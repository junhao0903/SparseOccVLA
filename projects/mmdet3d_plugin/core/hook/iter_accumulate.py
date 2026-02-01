# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class IterAccumulateHook(Hook):
    """Set runner's epoch information to the model."""

    def before_train_iter(self, runner):
        runner.model.module.iter = runner.iter