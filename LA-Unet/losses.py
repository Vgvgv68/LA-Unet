import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['BCEDiceLoss', 'LovaszHingeLoss', 'FocalBCEDiceLoss']


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice

class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss

class FocalBCEDiceLoss(nn.Module):
    def __init__(self,
                 alpha=0.75,
                 gamma=2.0,
                 smooth=1e-6,
                 focal_weight=0.7,
                 dice_weight=0.2,
                 iou_weight=0.1,
                 warmup_epochs=20):
        super().__init__()
        # 动态权重参数
        self.focal_weight = nn.Parameter(torch.tensor(focal_weight), requires_grad=False)
        self.dice_weight = nn.Parameter(torch.tensor(dice_weight), requires_grad=False)
        self.iou_weight = nn.Parameter(torch.tensor(iou_weight), requires_grad=False)

        # 动态权重调整
        self.warmup_scheduler = WarmupScheduler(
            total_epochs=warmup_epochs,
            initial_weights=(0.9, 0.05, 0.05),
            target_weights=(focal_weight, dice_weight, iou_weight)
        )

        # 基础参数
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, input, target, epoch=None):
        # Focal BCE计算
        bce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        probas = torch.sigmoid(input)
        focal_weight = self.alpha * (1 - probas) ** self.gamma * target + \
                       (1 - self.alpha) * probas ** self.gamma * (1 - target)
        focal_bce = (focal_weight * bce_loss).mean()

        # Dice计算
        input_flat = probas.view(-1)
        target_flat = target.view(-1)
        intersection = (input_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (input_flat.sum() + target_flat.sum() + self.smooth)
        dice_loss = 1 - dice

        # IoU计算
        union = input_flat.sum() + target_flat.sum() - intersection
        iou_loss = 1 - (intersection + self.smooth) / (union + self.smooth)

        # 动态权重调整
        if epoch is not None:
            fw, dw, iw = self.warmup_scheduler.get_weights(epoch)
        else:
            fw, dw, iw = self.focal_weight, self.dice_weight, self.iou_weight

        total_loss = fw * focal_bce + dw * dice_loss + iw * iou_loss

        return total_loss, {'focal_bce': focal_bce, 'dice': dice_loss, 'iou': iou_loss}


class WarmupScheduler:
    """动态权重调整策略"""

    def __init__(self, total_epochs, initial_weights, target_weights):
        self.total_epochs = total_epochs
        self.initial_weights = torch.tensor(initial_weights)
        self.target_weights = torch.tensor(target_weights)
        self.delta = (self.target_weights - self.initial_weights) / total_epochs

    def get_weights(self, epoch):
        progress = min(epoch, self.total_epochs)
        current = self.initial_weights + self.delta * progress
        return current[0].item(), current[1].item(), current[2].item()