from typing import List

import numpy as np
from typing import List

import torch

from sklearn.metrics import roc_curve, auc, average_precision_score
from sklearn.metrics import precision_score, recall_score, f1_score


def point_adjustment(y_true, y_score):

    score = y_score.clone()
    score = score.detach().cpu().numpy()
    assert len(score) == len(y_true)
    splits = np.where(y_true[1:] != y_true[:-1])[0] + 1
    is_anomaly = y_true[0] == 1
    pos = 0
    for sp in splits:
        if is_anomaly:
            score[pos:sp] = np.max(score[pos:sp])
        is_anomaly = not is_anomaly
        pos = sp
    sp = len(y_true)
    if is_anomaly:
        score[pos:sp] = np.max(score[pos:sp])
    return score


def test_performence(test_result:List[np.ndarray]) -> tuple:
    
    test_predictions, test_ground_truth, test_anomaly_label = test_result

    all_results = {}

    feature_dict = {}

    test_predictions = torch.tensor(test_predictions)

    test_predictions = [torch.tensor(x) for x in test_predictions]
    # test_predictions = F.softmax(test_predictions, dim=0)
    test_anomaly_label = [torch.tensor(x) for x in test_anomaly_label]

    test_predictions = torch.concat(test_predictions, dim=0)
    test_anomaly_label = torch.concat(test_anomaly_label, dim=0)

    roc_auc, prc_auc = test_roc_prc_perf(test_predictions, test_anomaly_label)
    all_results['roc_prc'] = {'roc_auc': roc_auc, 'prc_auc': prc_auc}
    print("Test ROC : {:.4f} PRC: {:.4f}".format(roc_auc, prc_auc))

    precision, recall, f1 = test_perf_based_on_best(test_predictions, test_anomaly_label)
    print("F1: {:.4f}; Test pre : {:.4f} recall: {:.4f} ".format(f1, precision, recall))

    
    return all_results, feature_dict, test_predictions, test_ground_truth, test_anomaly_label


def test_performence_pa(test_result:List[np.ndarray]) -> tuple:

    test_predictions, test_ground_truth, test_anomaly_label  = test_result

    all_results = {}
    feature_dict = {}

    feature_dict['label'] = test_anomaly_label

    test_predictions = torch.tensor(test_predictions)

    test_predictions = [torch.tensor(x) for x in test_predictions]
    # test_predictions = F.softmax(test_predictions, dim=0)
    test_anomaly_label = [torch.tensor(x) for x in test_anomaly_label]

    test_predictions = torch.concat(test_predictions, dim=0)
    test_anomaly_label = torch.concat(test_anomaly_label, dim=0)

    test_predictions_pa = point_adjustment(test_anomaly_label, test_predictions)

    roc_auc, prc_auc = test_roc_prc_perf(test_predictions_pa, test_anomaly_label)
    print(f'PA performance ....')
    all_results['roc_prc'] = {'roc_auc': roc_auc, 'prc_auc': prc_auc}
    print("Test ROC : {:.4f} PRC: {:.4f}".format(roc_auc, prc_auc))

    precision, recall, f1 = test_perf_based_on_best(test_predictions_pa, test_anomaly_label)
    print("F1: {:.4f}; Test pre : {:.4f} recall: {:.4f} ".format(f1, precision, recall))

    return all_results, feature_dict, test_predictions_pa, test_ground_truth, test_anomaly_label
   
    
def test_roc_prc_perf(test_scores, anomaly_labels):

    assert len(test_scores) == len(anomaly_labels)
    fpr, tpr, _ = roc_curve(anomaly_labels, test_scores)
    roc_auc = auc(fpr, tpr)
    prc_auc = average_precision_score(anomaly_labels, test_scores)
    return roc_auc, prc_auc


def test_perf_based_on_val(val_scores, test_scores, test_anomaly_label):
    threshold = np.max(val_scores)
    test_scores_peaks = np.max(test_scores, axis=0)
    predicted_label = (test_scores_peaks > threshold).astype(int)
    test_anomaly_label = test_anomaly_label.astype(int)
    
    assert predicted_label.shape == test_anomaly_label.shape
    
    precision = precision_score(test_anomaly_label, predicted_label)
    recall = recall_score(test_anomaly_label, predicted_label)
    f1 = f1_score(test_anomaly_label, predicted_label)
    
    return (precision, recall, f1)


def test_perf_based_on_best(test_scores, anomaly_labels, threshold_steps= 400) -> tuple:

    if isinstance(test_scores, torch.Tensor):
        test_scores = test_scores.detach().cpu().numpy()
    test_score = np.max(test_scores)
    min_score = np.min(test_scores)
    max_score = np.max(test_scores)
    best_f1 = 0
    best_threshold = 0

    for step in range(threshold_steps):
        threshold = min_score + (max_score - min_score) * step / threshold_steps

        predicted_labels = (test_scores > threshold).astype(int)  # init

        f1 = f1_score(anomaly_labels, predicted_labels)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    final_predicted_labels = (test_scores > best_threshold).astype(int)
    precision = precision_score(anomaly_labels, final_predicted_labels)
    recall = recall_score(anomaly_labels, final_predicted_labels)
    f1 = f1_score(anomaly_labels, final_predicted_labels)

    return (precision, recall, f1)
