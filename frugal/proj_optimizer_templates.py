import torch
from torch import nn
from torch.optim import Optimizer
from typing import Callable, List, Dict, Iterable
from abc import abstractmethod
import math

from .galore_projector import GaLoreProjector
from .coordinate_projector import CoordinateProjector


def prepare_proj_params(model, target_modules_list=None, proj_norms=False, proj_embeds=False, proj_logits=False):
    if target_modules_list is None:
        target_modules_list = ["attn", "mlp", 
                               "q_proj", "v_proj", "up_proj", "down_proj", "gate_proj", "k_proj", "o_proj",
                               "query", "value", "key", "intermediate.dense", "output.dense"]
    proj_params = []
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        if not any(target_key in module_name for target_key in target_modules_list):
            continue
        
        if module.weight.requires_grad:
            proj_params.append(module.weight)

    for name, p in model.named_parameters():
        if (("norm" in name and proj_norms) or 
            ("embed_tokens" in name and proj_embeds) or 
            ("lm_head" in name and proj_logits)):
            proj_params.append(p)

    id_proj_params = set(id(p) for p in proj_params)
    regular_params = [p for p in model.parameters() if id(p) not in id_proj_params and p.requires_grad]
    return [{'params': regular_params, 'is_proj_params': False}, 
            {'params': proj_params, 'is_proj_params': True}]


class ProjOptimizer(Optimizer):

    def __init__(
        self,

        params: Iterable[nn.parameter.Parameter],

        proj_params=None,

        proj_params_lr_scale = 1.0,
        update_gap: int = 200,
        density=0.25,
        reset_statistics=True, 
        inactive_update_rule='sign_sgd',
        inactive_lr_scale=1.0,

        _example_state_init=False,

        # Dynamic Rho parameters
        use_dynamic_rho=False,
        dynamic_rho_start=0.25,
        dynamic_rho_end=0.05,
        dynamic_rho_total_steps=200000,

        # Dynamic T parameters
        use_dynamic_t=False,
        dynamic_t_start_freq=100,
        dynamic_t_max_freq=1000,
        dynamic_t_eval_steps=5000,
        dynamic_t_loss_threshold_low=0.005,
        dynamic_t_increase_factor=1.5,
        dynamic_t_loss_for_increase_threshold=20.0,
    ):
        assert isinstance(params, List) or proj_params is not None, "One should be either seperate proj params in the 'params' or explicitly pass them as 'proj_params'."
        assert 0.0 <= density <= 1.0
        assert proj_params_lr_scale >= 0
        assert inactive_lr_scale >= 0

        # Dynamic Rho validation
        if use_dynamic_rho:
            assert 0.0 <= dynamic_rho_end <= dynamic_rho_start <= 1.0, "dynamic_rho_end must be less than or equal to dynamic_rho_start and both must be in [0.0, 1.0]"
            assert dynamic_rho_total_steps > 0, "dynamic_rho_total_steps must be positive"

        # Dynamic T validation
        if use_dynamic_t:
            assert dynamic_t_start_freq > 0, "dynamic_t_start_freq must be positive"
            assert dynamic_t_max_freq >= dynamic_t_start_freq, "dynamic_t_max_freq must be greater than or equal to dynamic_t_start_freq"
            assert dynamic_t_eval_steps > 0, "dynamic_t_eval_steps must be positive"
            assert dynamic_t_increase_factor > 1.0, "dynamic_t_increase_factor must be greater than 1.0"
        
        proj_params_args_dict = {
            'density': density, 
            'update_gap': update_gap, 
            'proj_params_lr_scale': proj_params_lr_scale, 
            'reset_statistics': reset_statistics, 
            'inactive_lr_scale': inactive_lr_scale, 
            'inactive_update_rule': inactive_update_rule, 
            '_example_state_init': _example_state_init,
            
            # Dynamic Rho parameters
            'use_dynamic_rho': use_dynamic_rho,
            'dynamic_rho_start': dynamic_rho_start,
            'dynamic_rho_end': dynamic_rho_end,
            'dynamic_rho_total_steps': dynamic_rho_total_steps,
            'current_rho': dynamic_rho_start,  # Initialize current_rho to start value
            'global_step': 0,  # Initialize global_step counter
            
            # Dynamic T parameters
            'use_dynamic_t': use_dynamic_t,
            'current_t_update_freq': dynamic_t_start_freq,
            'dynamic_t_max_freq': dynamic_t_max_freq,
            'dynamic_t_eval_steps': dynamic_t_eval_steps,
            'dynamic_t_loss_threshold_low': dynamic_t_loss_threshold_low,
            'dynamic_t_increase_factor': dynamic_t_increase_factor,
            'dynamic_t_loss_for_increase_threshold': dynamic_t_loss_for_increase_threshold,
            'steps_since_last_eval_for_t': 0,
            'previous_validation_loss_at_eval': float('inf'),
        }

        if not isinstance(params, List):
            id_proj_params = [id(p) for p in proj_params]
            regular_params = [p for p in params if id(p) not in id_proj_params]
            return [{'params': regular_params, 'is_proj_params': False},
                    {'params': proj_params, 'is_proj_params': True}.update(proj_params_args_dict)]
        else:
            for group in params:
                assert isinstance(group, Dict)
                if group.get("is_proj_params", False):
                    for k, v in proj_params_args_dict.items():
                        group.setdefault(k, v)
            return params


    def is_proj_group(self, group):
        return "is_proj_params" in group and group["is_proj_params"]


    @torch.no_grad()
    @abstractmethod
    def _update_states(self, group):
        pass


    @torch.no_grad()
    def _update_states_if_necessary(self, group):
        for p in group["params"]:
            if "step" in self.state[p] and self.state[p]["step"] % group["current_t_update_freq" if group.get("use_dynamic_t", False) else "update_gap"]:
                return
            else:
                step = self.state[p]["step"] if "step" in self.state[p] else 0
                break
        self._update_states(group)


    @torch.no_grad()
    @abstractmethod
    def _proj_params_update(self, grad, state, group):
        pass


    @torch.no_grad()
    def update_dynamic_rho(self, group):
        """
        Update the current_rho based on the global_step using linear decay.
        
        Args:
            group: The parameter group containing dynamic rho configuration
        
        Returns:
            Updated current_rho value
        """
        if not group.get("use_dynamic_rho", False):
            return group["density"]
        
        if group["dynamic_rho_total_steps"] <= 0 or group["dynamic_rho_start"] == group["dynamic_rho_end"]:
            return group["dynamic_rho_start"]
        
        # Calculate the current rho using linear decay
        progress = min(1.0, group["global_step"] / group["dynamic_rho_total_steps"])
        current_rho = max(
            group["dynamic_rho_end"],
            group["dynamic_rho_start"] - (group["dynamic_rho_start"] - group["dynamic_rho_end"]) * progress
        )
        
        # TODO (Dynamic Rho): Log current_rho, num_active_blocks/current_rank_r here.
        
        return current_rho


    @torch.no_grad()
    def update_t_freq_if_needed(self, validation_loss, group=None):
        """
        Update the T frequency based on validation loss if needed.
        
        Args:
            validation_loss: Current validation loss
            group: Optional specific parameter group to update. If None, updates all proj groups.
        """
        if group is not None:
            groups = [group]
        else:
            groups = [g for g in self.param_groups if self.is_proj_group(g) and g.get("use_dynamic_t", False)]
        
        for g in groups:
            if not g.get("use_dynamic_t", False):
                continue
                
            g["steps_since_last_eval_for_t"] += 1
            
            if g["steps_since_last_eval_for_t"] >= g["dynamic_t_eval_steps"]:
                # Reset counter
                g["steps_since_last_eval_for_t"] = 0
                
                # Calculate relative loss change
                prev_loss = g["previous_validation_loss_at_eval"]
                relative_loss_change = abs(validation_loss - prev_loss) / (prev_loss + 1e-9)
                
                # Update T frequency if loss change is small and loss is below threshold
                if (relative_loss_change < g["dynamic_t_loss_threshold_low"] and 
                    validation_loss < g["dynamic_t_loss_for_increase_threshold"]):
                    g["current_t_update_freq"] = min(
                        g["dynamic_t_max_freq"],
                        int(round(g["current_t_update_freq"] * g["dynamic_t_increase_factor"]))
                    )
                    
                # TODO (Dynamic T): Log current_T_update_freq, validation_loss, relative_loss_change here.
                
                # Update previous validation loss
                g["previous_validation_loss_at_eval"] = validation_loss


    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            if self.is_proj_group(group):
                # Increment global step counter
                group["global_step"] += 1
                
                # Update current_rho if dynamic rho is enabled
                if group.get("use_dynamic_rho", False):
                    group["current_rho"] = self.update_dynamic_rho(group)
                    
                    # If using BlockOptimizer, update num_active_blocks
                    if "num_active_blocks" in group:
                        old_num_active_blocks = group["num_active_blocks"]
                        total_blocks = len(group["params"])
                        new_num_blocks = int(math.floor(group["current_rho"] * total_blocks))
                        
                        # No need to ensure at least one active block
                        # Allow zero active blocks if that's what the calculation gives
                            
                        group["num_active_blocks"] = new_num_blocks
                        
                        # If num_active_blocks changed, we need to update subspace
                        if old_num_active_blocks != new_num_blocks:
                            # Force update of projection subspace
                            self._update_states(group)
                            
                self._update_states_if_necessary(group)
                
            for p in group["params"]:
                if p.grad is None:
                    continue
            
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("ProjOptimizer does not support sparse gradients")
                
                state = self.state[p]
                
                if len(state) == 0:
                    self._init_state(example=p, state=state)

                p.mul_(1 - group["lr"] * group["weight_decay"])

                state["step"] += 1

                if not self.is_proj_group(group):
                    update = self._compute_update(grad, state, **group)
                else:
                    update = self._proj_params_update(grad, state, group)
                

                p.add_(update)

        return loss
    

class GaloreOptimizer(ProjOptimizer):
    def __init__(
        self,

        params: Iterable[nn.parameter.Parameter],

        proj_params=None,

        proj_params_lr_scale = 1.0,
        update_gap: int = 200,
        density=0.25,
        reset_statistics=True,
        inactive_update_rule='sign_sgd',
        inactive_lr_scale=1.0,

        _example_state_init=False,

        # galore specific
        proj_side='std',
        proj_type='svd',

        # Dynamic Rho parameters
        use_dynamic_rho=False,
        dynamic_rho_start=0.25,
        dynamic_rho_end=0.05,
        dynamic_rho_total_steps=200000,

        # Dynamic T parameters
        use_dynamic_t=False,
        dynamic_t_start_freq=100,
        dynamic_t_max_freq=1000,
        dynamic_t_eval_steps=5000,
        dynamic_t_loss_threshold_low=0.005,
        dynamic_t_increase_factor=1.5,
        dynamic_t_loss_for_increase_threshold=20.0,
    ):
        params = super().__init__(
            params=params,
            proj_params=proj_params,
            proj_params_lr_scale=proj_params_lr_scale,
            update_gap=update_gap,
            density=density,
            reset_statistics=reset_statistics,
            inactive_update_rule=inactive_update_rule,
            inactive_lr_scale=inactive_lr_scale,
            _example_state_init=_example_state_init,
            use_dynamic_rho=use_dynamic_rho,
            dynamic_rho_start=dynamic_rho_start,
            dynamic_rho_end=dynamic_rho_end,
            dynamic_rho_total_steps=dynamic_rho_total_steps,
            use_dynamic_t=use_dynamic_t,
            dynamic_t_start_freq=dynamic_t_start_freq,
            dynamic_t_max_freq=dynamic_t_max_freq,
            dynamic_t_eval_steps=dynamic_t_eval_steps,
            dynamic_t_loss_threshold_low=dynamic_t_loss_threshold_low,
            dynamic_t_increase_factor=dynamic_t_increase_factor,
            dynamic_t_loss_for_increase_threshold=dynamic_t_loss_for_increase_threshold,
        )
        for group in params:
            if self.is_proj_group(group):
                group["proj_side"] = proj_side
                group["proj_type"] = proj_type
        return params
    

    @torch.no_grad()
    def _update_states(self, group):
        for p in group["params"]:
            state = self.state[p]
            grad = p.grad
            
            # Get current density based on dynamic or static settings
            current_density = group["current_rho"] if group.get("use_dynamic_rho", False) else group["density"]
            
            if "projector" not in state:
                state["projector"] = GaLoreProjector(current_density, grad_shape=grad.shape, proj_side=group["proj_side"], proj_type=group["proj_type"])
            elif group.get("use_dynamic_rho", False):
                # Update projector with new density if using dynamic rho
                state["projector"] = GaLoreProjector(current_density, grad_shape=grad.shape, proj_side=group["proj_side"], proj_type=group["proj_type"])
                
            state["projector"].update_proj(grad)
            if "step" not in state or group["reset_statistics"]:
                # reset
                if len(state) == 1 or group["_example_state_init"]:
                    self._init_state(example=state["projector"].project_down(grad), state=state)
                else:
                    self._init_state(state=state)

    
    @torch.no_grad()
    def _proj_params_update(self, grad, state, group):
        grad_down = state["projector"].project_down(grad)
        active_lr = group["lr"] * group["proj_params_lr_scale"]
        update = self._compute_update(grad_down, state, **{**group, 'lr': active_lr})
        update = state["projector"].project_up(update)
        if group["inactive_update_rule"] == "no":
            return update
        inactive_grad = grad - state["projector"].project_up(grad_down)
        inactive_lr = group["lr"] * group["proj_params_lr_scale"] * group["inactive_lr_scale"]
        if group["inactive_update_rule"] == "sgd":
            update.add_(-inactive_lr * inactive_grad)
        elif group["inactive_update_rule"] == "sign_sgd":
            update.add_(-inactive_lr * inactive_grad.sign())
        return update
    

class CoordOptimizer(ProjOptimizer):
    def __init__(
        self,

        params: Iterable[nn.parameter.Parameter],

        proj_params=None,

        proj_params_lr_scale = 1.0,
        update_gap: int = 200,
        density=0.25,
        reset_statistics=True,
        inactive_update_rule='sign_sgd',
        inactive_lr_scale=1.0,

        _example_state_init=False,
        
        # coord specific
        coord_choice='columns',

        # Dynamic Rho parameters
        use_dynamic_rho=False,
        dynamic_rho_start=0.25,
        dynamic_rho_end=0.05,
        dynamic_rho_total_steps=200000,

        # Dynamic T parameters
        use_dynamic_t=False,
        dynamic_t_start_freq=100,
        dynamic_t_max_freq=1000,
        dynamic_t_eval_steps=5000,
        dynamic_t_loss_threshold_low=0.005,
        dynamic_t_increase_factor=1.5,
        dynamic_t_loss_for_increase_threshold=20.0,
    ):
        params = super().__init__(
            params=params,
            proj_params=proj_params,
            proj_params_lr_scale=proj_params_lr_scale,
            update_gap=update_gap,
            density=density,
            reset_statistics=reset_statistics,
            inactive_update_rule=inactive_update_rule,
            inactive_lr_scale=inactive_lr_scale,
            _example_state_init=_example_state_init,
            use_dynamic_rho=use_dynamic_rho,
            dynamic_rho_start=dynamic_rho_start,
            dynamic_rho_end=dynamic_rho_end,
            dynamic_rho_total_steps=dynamic_rho_total_steps,
            use_dynamic_t=use_dynamic_t,
            dynamic_t_start_freq=dynamic_t_start_freq,
            dynamic_t_max_freq=dynamic_t_max_freq,
            dynamic_t_eval_steps=dynamic_t_eval_steps,
            dynamic_t_loss_threshold_low=dynamic_t_loss_threshold_low,
            dynamic_t_increase_factor=dynamic_t_increase_factor,
            dynamic_t_loss_for_increase_threshold=dynamic_t_loss_for_increase_threshold,
        )
        for group in params:
            if self.is_proj_group(group):
                group["coord_choice"] = coord_choice
        return params
    

    @torch.no_grad()
    def _update_states(self, group):
        for p in group["params"]:
            state = self.state[p]
            grad = p.grad
            
            # Get current density based on dynamic or static settings
            current_density = group["current_rho"] if group.get("use_dynamic_rho", False) else group["density"]
            
            if "projector" not in state:
                state["projector"] = CoordinateProjector(current_density, grad_shape=grad.shape, coord_choice=group["coord_choice"])
            elif group.get("use_dynamic_rho", False):
                # Update projector with new density if using dynamic rho
                state["projector"] = CoordinateProjector(current_density, grad_shape=grad.shape, coord_choice=group["coord_choice"])
                
            state["projector"].update_proj(grad)
            
            if "step" not in state or group["reset_statistics"]:
                # reset
                if len(state) == 1 or group["_example_state_init"]:
                    self._init_state(example=state["projector"].project_down(grad), state=state)
                else:
                    self._init_state(state=state)

    
    @torch.no_grad()
    def _proj_params_update(self, grad, state, group):
        grad_down = state["projector"].project_down(grad)
        active_lr = group["lr"] * group["proj_params_lr_scale"]
        update = self._compute_update(grad_down, state, **{**group, 'lr': active_lr})
        update = state["projector"].project_up(update)
        if group["inactive_update_rule"] == "no":
            return update
        inactive_grad = grad - state["projector"].project_up(grad_down)
        inactive_lr = group["lr"] * group["proj_params_lr_scale"] * group["inactive_lr_scale"]
        if group["inactive_update_rule"] == "sgd":
            update.add_(-inactive_lr * inactive_grad)
        elif group["inactive_update_rule"] == "sign_sgd":
            update.add_(-inactive_lr * inactive_grad.sign())
        return update
    

class BlockOptimizer(ProjOptimizer):
    def __init__(
        self,

        params: Iterable[nn.parameter.Parameter],

        proj_params=None,

        proj_params_lr_scale = 1.0,
        update_gap: int = 200,
        density=0.25,
        reset_statistics=True,
        inactive_update_rule='sign_sgd',
        inactive_lr_scale=1.0,

        _example_state_init=False,

        # block specific
        block_order='random',

        # Dynamic Rho parameters
        use_dynamic_rho=False,
        dynamic_rho_start=0.25,
        dynamic_rho_end=0.05,
        dynamic_rho_total_steps=200000,

        # Dynamic T parameters
        use_dynamic_t=False,
        dynamic_t_start_freq=100,
        dynamic_t_max_freq=1000,
        dynamic_t_eval_steps=5000,
        dynamic_t_loss_threshold_low=0.005,
        dynamic_t_increase_factor=1.5,
        dynamic_t_loss_for_increase_threshold=20.0,
    ):
        params = super().__init__(
            params=params,
            proj_params=proj_params,
            proj_params_lr_scale=proj_params_lr_scale,
            update_gap=update_gap,
            density=density,
            reset_statistics=reset_statistics,
            inactive_update_rule=inactive_update_rule,
            inactive_lr_scale=inactive_lr_scale,
            _example_state_init=_example_state_init,
            use_dynamic_rho=use_dynamic_rho,
            dynamic_rho_start=dynamic_rho_start,
            dynamic_rho_end=dynamic_rho_end,
            dynamic_rho_total_steps=dynamic_rho_total_steps,
            use_dynamic_t=use_dynamic_t,
            dynamic_t_start_freq=dynamic_t_start_freq,
            dynamic_t_max_freq=dynamic_t_max_freq,
            dynamic_t_eval_steps=dynamic_t_eval_steps,
            dynamic_t_loss_threshold_low=dynamic_t_loss_threshold_low,
            dynamic_t_increase_factor=dynamic_t_increase_factor,
            dynamic_t_loss_for_increase_threshold=dynamic_t_loss_for_increase_threshold,
        )
        for group in params:
            if self.is_proj_group(group):
                group["block_order"] = block_order
                
                # Calculate number of active blocks based on density or current_rho
                current_density = group.get("current_rho", density) if group.get("use_dynamic_rho", False) else density
                group["num_active_blocks"] = round(len(group["params"]) * current_density)
                
                assert not (group["block_order"] == "mirror" and group["num_active_blocks"] % 2), f"num tensors: {len(group['params'])}, num_active_blocks: {group['num_active_blocks']}"
        return params
    
    @torch.no_grad()
    def _activate_param(self, p):
        state = self.state[p]
        self._init_state(example=p, state=state)
        state["active"] = True

    @torch.no_grad()
    def _deactivate_param(self, p):
        state = self.state[p]
        state.clear()
        state["step"] = 0
        state["active"] = False

    @torch.no_grad()
    def _update_states(self, group):
        for p in group["params"]:
            # reset
            self._deactivate_param(p)
            
        # No need to ensure at least one active block
        # Allow zero active blocks if that's what dynamic rho gives us
            
        if group["block_order"] == "random":
            current_blocks = torch.randperm(len(group["params"]))[:group["num_active_blocks"]]
            for idx in current_blocks:
                self._activate_param(group["params"][idx])
        elif group["block_order"] == "ascending":
            if "next_block_start" not in group:
                group["next_block_start"] = 0
            for idx in range(group["next_block_start"], group["next_block_start"] + group["num_active_blocks"]):
                self._activate_param(group["params"][idx % len(group["params"])])
            group["next_block_start"] += group["num_active_blocks"]
            if group["next_block_start"] >= len(group["params"]):
                group["next_block_start"] -= len(group["params"])
        elif group["block_order"] == "descending":
            if "next_block_start" not in group:
                group["next_block_start"] = len(group["params"]) - 1
            for idx in range(group["next_block_start"], group["next_block_start"] - group["num_active_blocks"], -1):
                self._activate_param(group["params"][idx % len(group["params"])])
            group["next_block_start"] -= group["num_active_blocks"]
            if group["next_block_start"] < 0:
                group["next_block_start"] += len(group["params"])
        elif group["block_order"] == "mirror":
            if "next_block_start" not in group:
                group["next_block_start"] = 0
            for idx in range(group["next_block_start"], group["next_block_start"] + group["num_active_blocks"] // 2):
                self._activate_param(group["params"][idx % (len(group["params"]) // 2)])
                self._activate_param(group["params"][len(group["params"]) - 1 - idx % (len(group["params"]) // 2)])
            group["next_block_start"] += group["num_active_blocks"] // 2
            if group["next_block_start"] >= len(group["params"]) // 2:
                group["next_block_start"] -= len(group["params"]) // 2
    
    @torch.no_grad()
    def _proj_params_update(self, grad, state, group):
        if state["active"]:
            active_lr = group["lr"] * group["proj_params_lr_scale"]
            return self._compute_update(grad, state, **{**group, 'lr': active_lr})
        elif group["inactive_update_rule"] == "no":
            return torch.zeros_like(grad)
        inactive_lr = group["lr"] * group["proj_params_lr_scale"] * group["inactive_lr_scale"]
        if group["inactive_update_rule"] == "sgd":
            return -inactive_lr * grad
        elif group["inactive_update_rule"] == "sign_sgd":
            return -inactive_lr * grad.sign()