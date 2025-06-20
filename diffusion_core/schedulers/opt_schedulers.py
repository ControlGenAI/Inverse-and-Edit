import numpy as np
from diffusion_core.utils.class_registry import ClassRegistry


opt_registry = ClassRegistry()

@opt_registry.add_to_registry('constant')
class OptScheduler:
    def __init__(self, max_ddim_steps, max_inner_steps, early_stop_epsilon, plateau_prop):
        self.max_inner_steps = max_inner_steps
        self.max_ddim_steps = max_ddim_steps
        self.early_stop_epsilon = early_stop_epsilon
        self.plateau_prop = plateau_prop
        self.inner_steps_list = np.full(self.max_ddim_steps, self.max_inner_steps)

    def __call__(self, ddim_step, inner_step, loss=None):
        return inner_step + 1 >= self.inner_steps_list[ddim_step]
    

@opt_registry.add_to_registry('loss')
class LossOptScheduler(OptScheduler):
    def __call__(self, ddim_step, inner_step, loss):
        if loss < self.early_stop_epsilon + ddim_step * 2e-5:
            return True
        self.inner_steps_list[ddim_step] = inner_step + 1
        return False
    