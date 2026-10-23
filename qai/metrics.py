"""Metrics that mean something on imbalanced data (fraud): accuracy alone is useless there
(predicting "never fraud" scores 99.8% on the raw data)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

PRIMARY = {"default": "acc", "creditcard": "pr_auc"}


def classification_metrics(y_true, y_pred, p_pos=None) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    m = {"acc": float(np.mean(y_true == y_pred)),
         "bal_acc": float(balanced_accuracy_score(y_true, y_pred))}
    if len(np.unique(y_true)) == 2:
        m.update(precision=float(precision_score(y_true, y_pred, zero_division=0)),
                 recall=float(recall_score(y_true, y_pred, zero_division=0)),
                 f1=float(f1_score(y_true, y_pred, zero_division=0)))
        if p_pos is not None:
            m.update(roc_auc=float(roc_auc_score(y_true, p_pos)),
                     pr_auc=float(average_precision_score(y_true, p_pos)))
    return m


def softmax(logits):
    z = np.asarray(logits, float)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def from_logits(y_true, logits) -> dict:
    probs = softmax(logits)
    p_pos = probs[:, 1] if probs.shape[1] == 2 else None
    return classification_metrics(y_true, probs.argmax(1), p_pos)
