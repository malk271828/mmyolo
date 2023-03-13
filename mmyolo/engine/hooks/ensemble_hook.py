import torch

from mmengine.hooks import Hook
from mmyolo.registry import HOOKS

from ensemble_boxes import *

from rich import print

@HOOKS.register_module()
class EnsembleHook(Hook):
    """EnsembleHook.
    This hook will ...
    Args:
        None
    """

    def __init__(self, json_file):
        self.json_file = json_file
    
    def before_run(self, runner):
        pass

    def after_run(self, runner):
        print("after_run")

    def after_train_iter(self, runner):
        if self.every_n_iters(runner, self.interval):
            assert torch.isfinite(runner.outputs['loss']), \
                runner.logger.info('loss become infinite or NaN!')
