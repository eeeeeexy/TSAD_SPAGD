from typing import List

import numpy as np
from scipy.stats import iqr, rankdata
from typing import List

import torch

import torch.nn.functional as F

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score


def point_adjustment(y_true, y_score):
    """
    adjust the score for segment detection. i.e., for each ground-truth anomaly segment,
    use the maximum score as the score of all points in that segment. This corresponds to point-adjust f1-score.
    *This function is copied/modified from the source code in [Zhihan Li et al. KDD21]* 

    Args:
    
        y_true (np.array, required): 
            Data label, 0 indicates normal timestamp, and 1 is anomaly.
            
        y_score (np.array, required): 
            Predicted anomaly scores, higher score indicates higher likelihoods to be anomaly.

    Returns:
    
        np.array: 
            Adjusted anomaly scores.

    """
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

def calculate_and_smooth_error_scores(predictions: list or np.ndarray,
                                      ground_truth: list or np.ndarray,
                                      smoothing_window: int = 3,
                                      epsilon:float = 1e-2) -> np.ndarray:
    """
    Calculate and smooth the error scores between test predictions and ground truths.

    Args:
        predictions (list or np.ndarray): The predicted values on the test set.
        ground_truth (list or np.ndarray): The actual ground truth values of the test set.
        smoothing_window (int): The number of elements to consider for smoothing the error scores.
        epsilon (float): A small constant added for numerical stability.

    Returns:
        numpy.ndarray: Smoothed error scores.
    """
    
    test_delta = np.abs(np.array(predictions) - np.array(ground_truth))
    err_median = np.median(test_delta)
    err_iqr = iqr(test_delta)
    normalized_err_scores = (test_delta - err_median) / (np.abs(err_iqr) + epsilon)
    # smoothe the error scores by a moving average
    smoothed_err_scores = np.zeros_like(normalized_err_scores)
    for idx in range(smoothing_window, len(normalized_err_scores)):
        smoothed_err_scores[idx] = np.mean(normalized_err_scores[idx - smoothing_window: idx])
    return smoothed_err_scores

def calculate_nodewise_error_scores(predictions:np.ndarray,
                                    ground_truth:np.ndarray,
                                    smoothing_window:int=3,
                                    epsilon:float=1e-2) -> np.ndarray:
    # predictions: [total_time_len, num_nodes]
    # ground_truth: [total_time_len, num_nodes]
    # return: [num_nodes, total_time_len - smoothing_window + 1]
    nodewise_error_scores = []
    number_nodes = predictions.shape[1]
    for i in range(number_nodes):
        # import pdb; pdb.set_trace()
        pred = predictions[:, i]
        gt = ground_truth[:, i]
        scores = calculate_and_smooth_error_scores(pred, gt, smoothing_window, epsilon)
        nodewise_error_scores.append(scores)
    
    # [num_nodes, total_time_len - smoothing_window + 1]
    return np.stack(nodewise_error_scores, axis=0) 
    

def test_performence(test_result:List[np.ndarray]) -> tuple:
    """ Get the precision, recall and f1 score of the testset based on the validation set.

    Args:
        test_result (List): list of [test_predictions, test_ground_truth, test_anomaly_label]
        val_result (List): list of [val_predictions, val_ground_truth, val_anomaly_label]
    Returns:
        tuple: (precision, recall, f1)
    """
    
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

    # results = get_bestF1(test_predictions, test_anomaly_label)
    # print("new results :", results['AUC'], results['F1'])
    
    return all_results, feature_dict, test_predictions, test_ground_truth, test_anomaly_label


def test_performence_pa(test_result:List[np.ndarray]) -> tuple:
    """ Get the precision, recall and f1 score of the testset based on the validation set.

    Args:
        test_result (List): list of [test_predictions, test_ground_truth, test_anomaly_label]
        val_result (List): list of [val_predictions, val_ground_truth, val_anomaly_label]
    Returns:
        tuple: (precision, recall, f1)
    """
    
    test_predictions, test_ground_truth, test_anomaly_label  = test_result
    # val_predictions, val_ground_truth, _ = val_result
    # _predictions: [total_time_len, num_nodes]
    # _ground_truth: [total_time_len, num_nodes]
    # _anomaly_label: [total_time_len]
    # val_scores = calculate_nodewise_error_scores(val_predictions,
    #                                              val_ground_truth,
    #                                              smoothing_window,
    #                                              epsilon)
    # # test_scores: [num_nodes, total_time_len - smoothing_window + 1]
    # test_scores = calculate_nodewise_error_scores(test_predictions,
    #                                               test_ground_truth,
    #                                               smoothing_window,
    #                                               epsilon)
    all_results = {}

    feature_dict = {}

    # input_feature_final = np.concatenate(input_feature, axis=0)
    # output_feature_final = np.concatenate(output_feature, axis=0)
    # anomaly_label_final = np.concatenate(test_anomaly_label, axis=0)

    # feature_dict['inputdata'] = input_feature_final
    # feature_dict['outputdata'] = output_feature_final
    feature_dict['label'] = test_anomaly_label
    
    # all_results['best'] = {'precision': precision, 'recall': recall, 'f1': f1}
    # print("Test (best) Precision: {:.2f} Recall: {:.2f} F1: {:.4f}".format(precision*100, recall*100, f1))
    
    # precision, recall, f1 = test_perf_based_on_val(val_scores, test_scores, test_anomaly_label)
    # all_results['val'] = {'precision': precision, 'recall': recall, 'f1': f1}
    # print("Test (val) Precision: {:.2f} Recall: {:.2f} F1: {:.4f}".format(precision*100, recall*100, f1))

    # import pdb; pdb.set_trace()

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

    # results = get_bestF1(test_predictions, test_anomaly_label)
    # print("new results :", results['AUC'], results['F1'])
    
    return all_results, feature_dict, test_predictions_pa, test_ground_truth, test_anomaly_label


def test_performence_GDN_pa(test_result:List[np.ndarray],
                     smoothing_window:int=3,
                     epsilon:float=1e-2) -> tuple:
    """ Get the precision, recall and f1 score of the testset based on the validation set.

    Args:
        test_result (List): list of [test_predictions, test_ground_truth, test_anomaly_label]
        val_result (List): list of [val_predictions, val_ground_truth, val_anomaly_label]
    Returns:
        tuple: (precision, recall, f1)
    """
    
    test_predictions, test_ground_truth, test_anomaly_label = test_result

    test_scores = calculate_nodewise_error_scores(test_predictions,
                                                  test_ground_truth,
                                                  smoothing_window,
                                                  epsilon)
    all_results = {}

    test_scores = np.max(test_scores, axis=0)
    test_predictions_pa = point_adjustment(test_anomaly_label, torch.tensor(test_scores))
    
    roc_auc, prc_auc = test_roc_prc_perf_GDN(test_predictions_pa, test_anomaly_label, pa_flag=True)
    all_results['roc_prc'] = {'roc_auc': roc_auc, 'prc_auc': prc_auc}
    
    precision, recall, f1 = test_perf_based_on_best_GDN(test_predictions_pa, test_anomaly_label, pa_flag=True)
    all_results['best'] = {'precision': precision, 'recall': recall, 'f1': f1}

    print('PA formance ....')
    print("Test ROC : {:.4f} PRC: {:.4f} F1: {:.4f} Test (best) Precision: {:.2f} Recall: {:.2f} ".format(roc_auc, prc_auc, f1, precision*100, recall*100))
    
    return all_results
    


def test_performence_GDN(test_result:List[np.ndarray],
                     val_result:List[np.ndarray],
                     smoothing_window:int=3,
                     epsilon:float=1e-2) -> tuple:
    """ Get the precision, recall and f1 score of the testset based on the validation set.

    Args:
        test_result (List): list of [test_predictions, test_ground_truth, test_anomaly_label]
        val_result (List): list of [val_predictions, val_ground_truth, val_anomaly_label]
    Returns:
        tuple: (precision, recall, f1)
    """
    
    test_predictions, test_ground_truth, test_anomaly_label = test_result
    val_predictions, val_ground_truth, _ = val_result
    # _predictions: [total_time_len, num_nodes]
    # _ground_truth: [total_time_len, num_nodes]
    # _anomaly_label: [total_time_len]
    val_scores = calculate_nodewise_error_scores(val_predictions,
                                                 val_ground_truth,
                                                 smoothing_window,
                                                 epsilon)
    # test_scores: [num_nodes, total_time_len - smoothing_window + 1]
    test_scores = calculate_nodewise_error_scores(test_predictions,
                                                  test_ground_truth,
                                                  smoothing_window,
                                                  epsilon)
    all_results = {}
    
    # precision, recall, f1 = test_perf_based_on_val(val_scores, test_scores, test_anomaly_label)
    # all_results['val'] = {'precision': precision, 'recall': recall, 'f1': f1}
    # print("Test (val) Precision: {:.2f} Recall: {:.2f} F1: {:.4f}".format(precision*100, recall*100, f1))
    
    roc_auc, prc_auc = test_roc_prc_perf_GDN(test_scores, test_anomaly_label, pa_flag=False)
    all_results['roc_prc'] = {'roc_auc': roc_auc, 'prc_auc': prc_auc}

    precision, recall, f1 = test_perf_based_on_best_GDN(test_scores, test_anomaly_label, pa_flag=False)
    all_results['best'] = {'precision': precision, 'recall': recall, 'f1': f1}

    print("Test ROC : {:.4f} PRC: {:.4f} Test (best) Precision: {:.2f} Recall: {:.2f} F1: {:.4f}".format(roc_auc, prc_auc, precision*100, recall*100, f1))
    
    return all_results
    
    
    
def test_roc_prc_perf(test_scores, anomaly_labels):
    # test_scores: [num_nodes, total_time_len]
    # anomaly_labels: [total_time_len]

    # test_scores = np.max(test_scores, axis=0) # [time_len]
    assert len(test_scores) == len(anomaly_labels)
    fpr, tpr, _ = roc_curve(anomaly_labels, test_scores)
    roc_auc = auc(fpr, tpr)
    prc_auc = average_precision_score(anomaly_labels, test_scores)
    return roc_auc, prc_auc

def test_roc_prc_perf_GDN(test_scores, anomaly_labels, pa_flag):
    # test_scores: [num_nodes, total_time_len]
    # anomaly_labels: [total_time_len]
    if pa_flag == False:
        test_scores = np.max(test_scores, axis=0) # [time_len]
    fpr, tpr, _ = roc_curve(anomaly_labels, test_scores)
    roc_auc = auc(fpr, tpr)
    prc_auc = average_precision_score(anomaly_labels, test_scores)
    return roc_auc, prc_auc


def test_perf_based_on_val(val_scores, test_scores, test_anomaly_label):
    # val_scores: [num_nodes, total_time_len]
    # test_scores: [num_nodes, total_time_len]
    # test_anomaly_label: [total_time_len]
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
    # find the best threshold based on the f1 score of the test set

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
        # average_pred = test_scores.mean(axis=0)
        # predicted_labels = (average_pred > 0.7).astype(int)

        # import pdb; pdb.set_trace()

        f1 = f1_score(anomaly_labels, predicted_labels)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    # average_pred = test_scores.mean(axis=0)
    # final_predicted_labels = (average_pred > best_threshold).astype(int)

    final_predicted_labels = (test_scores > best_threshold).astype(int)
    precision = precision_score(anomaly_labels, final_predicted_labels)
    recall = recall_score(anomaly_labels, final_predicted_labels)
    f1 = f1_score(anomaly_labels, final_predicted_labels)

    return (precision, recall, f1)


def test_perf_based_on_best_GDN(test_scores, anomaly_labels, pa_flag, threshold_steps= 400) -> tuple:
    # find the best threshold based on the f1 score of the test set
    if pa_flag == False:
        test_scores = np.max(test_scores, axis=0)
    min_score = np.min(test_scores)
    max_score = np.max(test_scores)
    best_f1 = 0
    best_threshold = 0

    for step in range(threshold_steps):
        threshold = min_score + (max_score - min_score) * step / threshold_steps
        predicted_labels = (test_scores > threshold).astype(int)
        f1 = f1_score(anomaly_labels, predicted_labels)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    final_predicted_labels = (test_scores > best_threshold).astype(int)
    precision = precision_score(anomaly_labels, final_predicted_labels)
    recall = recall_score(anomaly_labels, final_predicted_labels)
    f1 = f1_score(anomaly_labels, final_predicted_labels)

    return (precision, recall, f1)


def get_bestF1(scores, lab, PA=False):
    scores = scores.numpy() if torch.is_tensor(scores) else scores
    lab = lab.numpy() if torch.is_tensor(lab) else lab
    ones = lab.sum()
    zeros = len(lab) - ones
    
    sortid = np.argsort(scores - lab * 1e-16)
    new_lab = lab[sortid]
    new_scores = scores[sortid]
    
    if PA:
        lab_diff = np.insert(lab, len(lab), 0) - np.insert(lab, 0, 0)
        a_st = np.arange(len(lab)+1)[lab_diff == 1]
        a_ed = np.arange(len(lab)+1)[lab_diff == -1]

        thres_a = np.array([np.max(scores[a_st[i]:a_ed[i]]) for i in range(len(a_st))])
        sort_a_id = np.flip(np.argsort(thres_a)) # big to small
        cum_a = np.cumsum(a_ed[sort_a_id] - a_st[sort_a_id])

        last_thres = np.inf
        TPs = np.zeros_like(new_lab)
        for i, a_id in enumerate(sort_a_id):
            TPs[(thres_a[a_id] <= new_scores) & (new_scores < last_thres)] = cum_a[i-1] if i > 0 else 0
            last_thres = thres_a[a_id]
        TPs[new_scores < last_thres] = cum_a[-1]
    else:
        TPs = np.cumsum(-new_lab) + ones
        
    FPs = np.cumsum(new_lab-1) + zeros
    FNs = ones - TPs
    TNs = zeros - FPs
    
    N = len(lab) - np.flip(TPs > 0).argmax()
    TPRs = TPs[:N] / ones
    PPVs = TPs[:N] / (TPs + FPs)[:N]
    FPRs = FPs[:N] / zeros
    F1s  = 2 * TPRs * PPVs / (TPRs + PPVs)
    maxid = np.argmax(F1s)
    
    FPRs = np.insert(FPRs, -1, 0)
    TPRs = np.insert(TPRs, -1, 0)
    if PA:
        AUC = ((TPRs[:-1] + TPRs[1:]) * (FPRs[:-1] - FPRs[1:])).sum() * 0.5
    else:
        import pdb; pdb.set_trace()
        AUC = roc_auc_score(lab, scores)
   
    anomaly_ratio = ones / len(lab) 
    FPR_bestF1_TPR1 = anomaly_ratio / (1-anomaly_ratio) * (2 / F1s[maxid] - 2)
    TPR_bestF1_FPR0 = F1s[maxid] / (2 - F1s[maxid])
    return {'AUC': AUC, 'F1': F1s[maxid], 'thres': new_scores[maxid], 'TPR': TPRs[maxid], 'PPV': PPVs[maxid], 
            'FPR': FPRs[maxid], 'maxid': maxid, 'FPRs': FPRs, 'TPRs': TPRs, 
            'FPR_bestF1_TPR1': FPR_bestF1_TPR1, 'TPR_bestF1_FPR0': TPR_bestF1_FPR0}
