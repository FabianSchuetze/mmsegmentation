import torch
import torch.nn as nn

from ..builder import LOSSES

@LOSSES.register_module
class MyLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0, loss_name='loss_mse'):
        super(MyLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self._loss = torch.nn.MSELoss(reduction='mean')
        self._loss_name = loss_name

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                ignore_index=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        target = target / 255.0
        bs, w, h = pred.size(0), pred.size(2), pred.size(3)
        target = target.reshape(bs,1, w, h)
        loss = self.loss_weight * self._loss(pred, target)
        return loss

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
