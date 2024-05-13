import scipy
import numpy as np
import pandas as pd
import copy
from src.names import *


def delete_low_and_binarize(df, delete_low=False):
    new_df = df.copy()
    if delete_low == True:
        new_df = new_df[(new_df[RATING] == 4) | (new_df[RATING] == 5)]
    new_df[RATING] = 1
    return new_df


def get_matrix(df, total_users, total_items):
    if total_users is None and total_items is None:
        total_users = max(df[USERID]) + 1
        total_items = max(df[ITEMID]) + 1
    assert max(df[USERID]) < total_users
    assert max(df[ITEMID]) < total_items
    return scipy.sparse.coo_matrix(
        (df[RATING], (df[USERID], df[ITEMID])), 
        shape=(total_users, total_items),
        dtype=np.float32
        ).tocsr()


def get_unique_recomendations(predicted, K):
    return np.unique(np.array(predicted[:, :K]).ravel())


def calculate_stability(predictions1, ids1, predictions2, ids2):
    sim = 0.
    
    ids1 = ids1[:len(predictions1)]
    ids2 = ids2[:len(predictions2)]
    mask_intersection_in_2 = np.in1d(ids2, ids1)
    mask_intersection_in_1 = np.in1d(ids1, ids2[mask_intersection_in_2])

    cut_predictions1 = predictions1[mask_intersection_in_1]
    cut_predictions2 = predictions2[mask_intersection_in_2]
    assert len(cut_predictions1) == len(cut_predictions2)

    for pred1, pred2 in zip(cut_predictions1, cut_predictions2):
        all_predictions = np.unique([pred1, pred2])
        numerator = 0.
        denumerator = 0.
        for item in all_predictions:
            r1 = np.where(pred1 == item)[0]
            r2 = np.where(pred2 == item)[0]
            w1 = 0.
            w2 = 0.
            if len(r1) > 0:
                w1 = 1 / (r1[0] + 1)
            if len(r2) > 0:
                w2 = 1 / (r2[0] + 1)
            numerator += min(w1, w2)
            denumerator += max(w1, w2)
        sim += numerator / denumerator
    return sim / len(cut_predictions1)


def calculate_metrics(K, predicted, actual, total_items, cur_predictions,
                    cur_ids,
                    prev_predictions,
                    prev_ids, n_test_users=None):
    # hits_mask = np.in1d(predicted, actual).reshape(predicted.shape)
    hits_mask = (predicted[:, :K] == actual)
    if n_test_users is None:
        n_test_users = actual.shape[0]
    # HR calculation
    hr = np.sum(hits_mask.any(axis=1)) / n_test_users
    # hr = np.mean(hits_mask.any(axis=1))

    # MRR calculation
    hit_rank = np.where(hits_mask)[1] + 1.0
    mrr = np.sum(1 / hit_rank) / n_test_users

    # NDCG calculation
    ndcg = np.sum(1 / np.log2(hit_rank + 1)) / n_test_users

    # Coverage calculation
    coverage = len(get_unique_recomendations(predicted[:, :K], K)) / total_items

    # Stability calculation
    stabitily = 0.
    if prev_predictions is not None:
        stabitily = calculate_stability(
            cur_predictions[:, :K],
            cur_ids,
            prev_predictions[:, :K],
            prev_ids
        )

    return hr, mrr, ndcg, coverage, stabitily


def create_virtual_users(A, next_portion):
    A_copy = A.copy()
    A_total = None
    users_actual_watched_total = None
    next_portion = next_portion.sort_values(by=[USERID, TIMESTAMP, ITEMID])   
    all_user_ids = np.array([])

    i = 0
    while len(next_portion) > 0:
        if i % 100 == 0:
            print('iter', i)
        i += 1

        test_user_x_first_item = next_portion.groupby(USERID).head(1).sort_values(by=[USERID])
        users_ids = test_user_x_first_item[USERID]
        all_user_ids = np.concatenate([all_user_ids, users_ids])

        A_act = A_copy[users_ids]
        users_actual_watched = test_user_x_first_item[ITEMID].to_numpy().reshape(-1, 1)

        if A_total is None:
            A_total = A_act.copy()
            users_actual_watched_total = users_actual_watched.copy()
        else:
            users_actual_watched_total = np.vstack((users_actual_watched_total, users_actual_watched))
            A_total = scipy.sparse.vstack((A_total, A_act))

        matrix_update = get_matrix(test_user_x_first_item, A_copy.shape[0], A_copy.shape[1])
        A_copy += matrix_update
        
        next_portion = next_portion[next_portion.groupby(USERID).cumcount() != 0]

    return A_total, users_actual_watched_total, all_user_ids


def collect_dirt(next_portion, total_users, total_items, userid = USERID, itemid=ITEMID):
    clean = next_portion[(next_portion[userid] < total_users)
                                & (next_portion[itemid] < total_items)]
    dirt = next_portion[(next_portion[userid] >= total_users) 
                                | (next_portion[itemid] >= total_items)]
    assert (clean.shape[0] + dirt.shape[0]) == next_portion.shape[0]
    return clean, dirt


def half_time_split(portion, timeid=TIMESTAMP):
    min_time = np.min(portion[timeid])
    max_time = np.max(portion[timeid])
    interval = (max_time - min_time) / 2
    mid_time = min_time + interval
    portion12 = portion[portion[timeid] < mid_time]
    return portion12


def update_mapping(dirt, column_name):
    '''
    сквозная перенумерация айдишников по возрастанию
    '''
    original_index_order = dirt[column_name].drop_duplicates(keep='first').to_numpy()
    assert len(original_index_order) == len(np.unique(dirt[column_name]))
    new_mapping = dict(zip(
        original_index_order,
        list(range(len(original_index_order))))
    )
    return new_mapping


def apply_mapping(data, mapping, column_name, inplace=True):
    return data.replace({column_name: mapping}, inplace=inplace)


def pure_cold_svd_update(U, S, Vt, cold_ui_matrix, r):
    U_small, S_small, Vt_small = scipy.sparse.linalg.svds(cold_ui_matrix, k=r)

    U = np.concatenate((U, U_small), axis=0)

    if len(S.shape) != 1:
        S = np.diag(S)
    S = np.sort(np.concatenate(S, S_small))[::-1]

    Vt = np.concatenate((Vt, Vt_small), axis=1)
    
    return U, S, Vt
