# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.
import warnings


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k.
    Deprecated.

    Args:
        output (torch.Tensor): Normally, output is classifier logits output with size (N, C),
            N is batch_size, C is num_classes.
        target (torch.Tensor): Normally, target is a tensor with size (N, ).
        topk (tuple, int):

    Returns:
        A list contains accuracy scalar tensor by topk
    """
    warnings.warn("essmc2.utils.metric.accuracy function is deprecated. "
                  "Use essmc2.utils.metrics.AccuracyMetric instead.")
    if isinstance(topk, int):
        topk = (topk,)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res
