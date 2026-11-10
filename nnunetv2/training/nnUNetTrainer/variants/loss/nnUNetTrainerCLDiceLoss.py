import numpy as np
import torch

from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss, DC_and_CE_loss, Custom_compound_scheduled_loss
from nnunetv2.training.loss.cldice import SoftDiceCLDiceLoss, SoftCLDiceLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import softmax_helper_dim1
from nnunetv2.experiment_planning.experiment_planners.residual_unets.residual_encoder_unet_planners import ResEncUNetPlanner


class nnUNetTrainerCLDiceLoss(nnUNetTrainer):
    def _build_loss(self):
        loss = SoftDiceCLDiceLoss()

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss



class nnUNetTrainerScheduledCLDiceLoss(nnUNetTrainer):
    decay_type = 'linear'
    
    def _build_loss(self):
        loss_1 = MemoryEfficientSoftDiceLoss(**{'batch_dice': self.configuration_manager.batch_dice,
            'do_bg': self.label_manager.has_regions, 'smooth': 1e-5, 'ddp': self.is_ddp},
            apply_nonlin=torch.sigmoid if self.label_manager.has_regions else softmax_helper_dim1)

        loss_2 = SoftCLDiceLoss(iter_=3) # k=5~25 in original paper, we use 3 to test for memory efficiency

        loss = Custom_compound_scheduled_loss(
                loss_1, loss_2, initial_alpha=1.0, decay_factor=0.002, 
                max_epoch=self.num_epochs, decay_type=self.decay_type)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        
        return loss

    def on_epoch_start(self):
        super().on_epoch_start()

        scheduled_loss = self.loss
        if isinstance(scheduled_loss, DeepSupervisionWrapper):
            scheduled_loss = scheduled_loss.loss

        if hasattr(scheduled_loss, 'set_epoch'):
            scheduled_loss.set_epoch(self.current_epoch)
            self.print_to_log_file(
                f'compound loss alpha: {scheduled_loss.alpha:.4f}'
            )

class nnUNetTrainerLinearScheduledCLDiceLoss(nnUNetTrainerScheduledCLDiceLoss):
    decay_type = 'linear'

class nnUNetTrainerExponentialScheduledCLDiceLoss(nnUNetTrainerScheduledCLDiceLoss):
    decay_type = 'exponential'