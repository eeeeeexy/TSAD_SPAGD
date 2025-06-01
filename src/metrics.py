from sklearn.metrics import auc, precision_recall_curve, precision_recall_fscore_support, roc_auc_score

def compute_auprc_baseline(target, score): # anomely: 1: normal: 0
    precision, recall, thresholds = precision_recall_curve(target, -1*score)
    auprc = auc(recall, precision)
    return auprc


def compute_auprc(target, score): # anomely: 0; normal: 1
    precision, recall, thresholds = precision_recall_curve(target, score)
    auprc = auc(recall, precision)
    return auprc


import numpy as np
def compute_pre_recall_f1_baseline(target, score):
    normal_ratio = (target == 0).sum() / len(target)
    threshold = np.percentile(score, 100 * normal_ratio)
    pred = np.zeros(len(score))
    pred[score > threshold] = 1

    precision, recall_, f1, _ = precision_recall_fscore_support(target, pred, average='binary')
    return f1, recall_, threshold


def compute_pre_recall_f1(target, score):
    normal_ratio = (target == 1).sum() / len(target)
    threshold = np.percentile(score, 100-100 * normal_ratio)
    pred = np.zeros(len(score))
    pred[score > threshold] = 1

    precision, recall_, f1, _ = precision_recall_fscore_support(target, pred, average='binary')
    return f1, recall_, threshold


def compute_roc_auc_score(target, score):
    roc_auc = roc_auc_score(target, score)
    return roc_auc