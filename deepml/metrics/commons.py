import numpy as np
from sklearn.metrics import confusion_matrix


def true_positives(output, target):
    return (output * target).sum()


def false_positives(output, target):
    return (output * (1 - target)).sum()


def false_negatives(output, target):
    return ((1 - output) * target).sum()


def true_negatives(output, target):
    return ((1 - output) * (1 - target)).sum()


def multiclass_tp_fp_tn_fn(output, target):
    cf_matrix = confusion_matrix(target.cpu(), output.cpu())
    tp = np.diag(cf_matrix)
    fp = cf_matrix.sum(axis=0) - tp
    fn = cf_matrix.sum(axis=1) - tp
    tn = cf_matrix.sum() - (fp + fn + tp)
    return tp.sum(), fp.sum(), tn.sum(), fn.sum()
