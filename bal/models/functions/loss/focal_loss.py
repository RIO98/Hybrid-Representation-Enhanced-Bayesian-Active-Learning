import torch
import torch.nn.functional as F


def focal_loss(inputs: torch.Tensor, targets: torch.Tensor, weight: torch.Tensor = None, gamma: float = 2.0,
               reduction: str = 'mean') -> torch.Tensor:
    """
    Compute the focal loss between inputs (logits) and targets (labels) for multi-class classification, based
    on the paper "Focal Loss for Dense Object Detection" (https://arxiv.org/abs/1708.02002).

    :param inputs: The input tensor (logits) of shape (batch_size, num_classes).
    :type inputs: torch.Tensor
    :param targets: The target tensor (labels) of shape (batch_size,).
    :type targets: torch.Tensor
    :param weight: A manual rescaling weight. Default: None.
    :type weight: torch.Tensor or None
    :param gamma: The focusing parameter. Default: 2.0.
    :type gamma: float
    :param reduction: The reduction method to apply. Can be 'mean', 'sum', or 'none'. Default: 'mean'.
    :type reduction: str
    :return: The focal loss.
    :rtype: torch.Tensor
    """

    ce_loss = F.cross_entropy(inputs, targets, weight=weight, reduction=reduction)
    pt = torch.exp(-ce_loss)
    if reduction == 'sum':
        floss = ((1 - pt) ** gamma * ce_loss).mean()
    else:
        floss = ((1 - pt) ** gamma * ce_loss).sum(dim=(-2, -1))
    return floss
