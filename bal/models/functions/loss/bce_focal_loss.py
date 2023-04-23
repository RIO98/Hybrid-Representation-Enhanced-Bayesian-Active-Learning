import torch
import torch.nn.functional as F


def bce_focal_loss(inputs: torch.Tensor,
                   targets: torch.Tensor,
                   alpha: float = 0.25,
                   gamma: float = 2,
                   reduction: str = 'none') -> torch.Tensor:
    """
    Compute the binary focal loss between inputs (logits) and targets (labels), based on the paper
    "Focal Loss for Dense Object Detection" (https://arxiv.org/abs/1708.02002).

    :param inputs: The input tensor (logits).
    :type inputs: torch.Tensor
    :param targets: The target tensor (labels).
    :type targets: torch.Tensor
    :param alpha: A balancing factor for the positive class. Default: 0.25.
    :type alpha: float
    :param gamma: The focusing parameter. Default: 2.
    :type gamma: float
    :param reduction: The reduction method to apply. Can be 'mean', 'sum', or 'none'. Default: 'none'.
    :type reduction: str
    :return: The binary focal loss.
    :rtype: torch.Tensor
    """

    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss
