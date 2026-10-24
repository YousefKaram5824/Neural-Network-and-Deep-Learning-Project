import numpy as np


def flat(x):
    return np.asarray(x).ravel()


def div(num, den):
    return num / den if den else 0.0


def count(y_true, y_pred, positive, negative):
    res = conf_mat(y_true, y_pred, positive, negative)
    tn, fp = res[0]
    fn, tp = res[1]
    return tn, fp, fn, tp


def conf_mat(y_true, y_pred, positive=1, negative=-1):
    y_true = flat(y_true)
    y_pred = flat(y_pred)

    tp = int(np.sum((y_true == positive) & (y_pred == positive)))
    tn = int(np.sum((y_true == negative) & (y_pred == negative)))
    fp = int(np.sum((y_true == negative) & (y_pred == positive)))
    fn = int(np.sum((y_true == positive) & (y_pred == negative)))

    return [[tn, fp], [fn, tp]]


def acc(y_true, y_pred):
    y_true = flat(y_true)
    y_pred = flat(y_pred)
    return float(np.mean(y_true == y_pred)) if y_true.size else 0.0


def pre(y_true, y_pred, positive=1, negative=-1):
    _, fp, _, tp = count(y_true, y_pred, positive, negative)
    return div(tp, tp + fp)


def re(y_true, y_pred, positive=1, negative=-1):
    _, _, fn, tp = count(y_true, y_pred, positive, negative)
    return div(tp, tp + fn)


def f1(y_true, y_pred, positive=1, negative=-1):
    p = pre(y_true, y_pred, positive, negative)
    r = re(y_true, y_pred, positive, negative)
    return div(2 * p * r, p + r)
