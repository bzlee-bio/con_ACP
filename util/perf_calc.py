import torch


def conf_matrix_calc(logit: torch.Tensor, y_batch: torch.Tensor):
    """Return confusion matrix

    Args:
        logit (Tensor): Logit/softmax output, (batch, n_cls)
        y_batch (Tensor): (batch,)

    Returns:
        Tensor: confusion matrix
    """

    _, logit = torch.max(logit, 1)
    y_batch = torch.squeeze(y_batch)
    conf_mat = torch.zeros(2, 2)
    tp = ((logit == 1) & (y_batch == 1)).sum().item()
    tn = ((logit == 0) & (y_batch == 0)).sum().item()
    fp = ((logit == 1) & (y_batch == 0)).sum().item()
    fn = ((logit == 0) & (y_batch == 1)).sum().item()
    conf_mat[0, 0] = tn
    conf_mat[0, 1] = fp
    conf_mat[1, 0] = fn
    conf_mat[1, 1] = tp
    return conf_mat


def metric_calc(conf_mat, return_conf=False):
    tn = conf_mat[0, 0].item()
    tp = conf_mat[1, 1].item()
    fp = conf_mat[0, 1].item()
    fn = conf_mat[1, 0].item()
    acc = (tn + tp) / (tn + tp + fp + fn)
    sn = tp / (tp + fn)
    sp = tn / (fp + tn)
    mcc_base = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if mcc_base != 0:
        mcc = (tp * tn - fp * fn) / (
            (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        ) ** 0.5
    else:
        mcc = -999
    res = {"acc": acc, "sn": sn, "sp": sp, "mcc": mcc}
    if return_conf:
        res["tn"] = tn
        res["tp"] = tp
        res["fp"] = fp
        res["fn"] = fn
    return res
