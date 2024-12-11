from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class CustomSetEpochInfoHook(Hook):
    """Set runner's epoch information to the model."""

    def before_train_epoch(self, runner):
        epoch = runner.epoch
        # 判断是否是CBGSDataset
        if runner.data_loader.dataset.__class__.__name__ == 'CBGSDataset':
            dataset = runner.data_loader.dataset.dataset
        else:
            dataset = runner.data_loader.dataset
        
        dataset.set_epoch(epoch)