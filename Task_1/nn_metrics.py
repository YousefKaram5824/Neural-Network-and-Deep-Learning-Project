"""
Evaluation metrics (implemented from scratch - sklearn is NOT allowed).

Convention:
    +1  -> positive class
    -1  -> negative class
"""

import numpy as np


def confusion_matrix(y_true, y_pred, positive_label=1, negative_label=-1):
    """
    Build a 2x2 confusion matrix.

    Returns
    -------
    cm : list of lists
        [[TN, FP],
         [FN, TP]]
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    TP = TN = FP = FN = 0
    for t, p in zip(y_true, y_pred):
        if t == positive_label and p == positive_label:
            TP += 1
        elif t == negative_label and p == negative_label:
            TN += 1
        elif t == negative_label and p == positive_label:
            FP += 1
        elif t == positive_label and p == negative_label:
            FN += 1

    return [[TN, FP], [FN, TP]]


def accuracy(y_true, y_pred):
    """Overall accuracy = (TP + TN) / total."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def precision(y_true, y_pred, positive_label=1, negative_label=-1):
    cm = confusion_matrix(y_true, y_pred, positive_label, negative_label)
    TN, FP = cm[0]
    FN, TP = cm[1]
    return TP / (TP + FP) if (TP + FP) > 0 else 0.0


def recall(y_true, y_pred, positive_label=1, negative_label=-1):
    cm = confusion_matrix(y_true, y_pred, positive_label, negative_label)
    TN, FP = cm[0]
    FN, TP = cm[1]
    return TP / (TP + FN) if (TP + FN) > 0 else 0.0


def f1_score(y_true, y_pred, positive_label=1, negative_label=-1):
    p = precision(y_true, y_pred, positive_label, negative_label)
    r = recall(y_true, y_pred, positive_label, negative_label)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
