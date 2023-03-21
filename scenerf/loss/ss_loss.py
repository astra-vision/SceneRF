import torch


def compute_l1_loss(pred, target, predicted_depth=None):
    """
    pred: (3, B)
    target: (3, B)
    ---
    return
    l1_loss: (B,)

    """
    abs_diff = torch.abs(target - pred)
    if predicted_depth is not None:
        abs_diff = abs_diff[:, predicted_depth < 30]
    l1_loss = abs_diff.mean(0)

    return l1_loss

