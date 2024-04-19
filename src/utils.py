import scipy
import numpy as np
import pandas as pd
import copy


def delete_low_and_binarize(df):
    new_df = copy.copy(df[(df['rating'] == 4) | (df['rating'] == 5)])
    new_df['rating'] = 1
    return new_df


def get_matrix(df, total_users, total_movies):
    assert max(df['userid']) < total_users
    assert max(df['movieid']) < total_movies
    matrix = df.pivot(index = 'userid', columns ='movieid', values = 'rating').fillna(0)
    matrix = matrix.reindex(columns=range(0, total_movies), fill_value=0)
    matrix = matrix.reindex(index=range(0, total_users), fill_value=0)
    return scipy.sparse.csr_matrix(matrix.values)


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


def calculate_metrics(K, predicted, actual, total_movies, cur_predictions,
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
    # Coverage calculation
    coverage = len(get_unique_recomendations(predicted[:, :K], K)) / total_movies

    # Stability calculation
    stabitily = 0.
    if prev_predictions is not None:
        stabitily = calculate_stability(
            cur_predictions[:, :K],
            cur_ids,
            prev_predictions[:, :K],
            prev_ids
        )

    return hr, mrr, coverage, stabitily


def create_virtual_users(A, next_portion):
    A_copy = A.copy()
    A_total = None
    users_actual_watched_total = None
    next_portion = next_portion.sort_values(by=['userid', 'timestamp', 'movieid'])   
    all_user_ids = np.array([])

    i = 0
    while len(next_portion) > 0:
        if i % 50 == 0:
            print('iter', i)
        i += 1

        test_user_x_first_item = next_portion.groupby('userid').head(1).sort_values(by=['userid'])
        users_ids = test_user_x_first_item['userid']
        all_user_ids = np.concatenate([all_user_ids, users_ids])

        A_act = A_copy[users_ids]
        users_actual_watched = test_user_x_first_item['movieid'].to_numpy().reshape(-1, 1)

        if A_total is None:
            A_total = A_act.copy()
            users_actual_watched_total = users_actual_watched.copy()
        else:
            users_actual_watched_total = np.vstack((users_actual_watched_total, users_actual_watched))
            A_total = scipy.sparse.vstack((A_total, A_act))

        matrix_update = get_matrix(test_user_x_first_item, A_copy.shape[0], A_copy.shape[1])
        A_copy += matrix_update
        
        next_portion = next_portion[next_portion.groupby('userid').cumcount() != 0]

    return A_total, users_actual_watched_total, all_user_ids


def collect_dirt(next_portion, total_users, total_movies, userid = 'userid', movieid='movieid'):
    clean = next_portion[(next_portion[userid] < total_users)
                                & (next_portion[movieid] < total_movies)]
    dirt = next_portion[(next_portion[userid] >= total_users) 
                                | (next_portion[movieid] >= total_movies)]
    assert (clean.shape[0] + dirt.shape[0]) == next_portion.shape[0]
    return clean, dirt


def half_time_split(portion, timeid='timestamp'):
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


def reorthogonalization(Q, tol=1e-14):
    if np.abs(Q[:, -1].T @ Q[:, -1]) > tol:
        for i in range(Q.shape[1]):
            alpha = Q[:, i]
            for j in range(i - 1):
                mult = alpha.T @ Q[:, j]
                Q[:, i] -= (Q[:, j].reshape(-1, 1) * mult).reshape(Q[:, i].shape)
            norm = np.linalg.norm(Q[:, i])
            Q[:, i] /= norm
    return Q


def matthew_brand(C, U, S, Vt, r, update_rows=False, tol=1e-14):
    '''
    C - columns to edit
    '''

    if update_rows:
        C = C.T
        U, S, Vt = Vt.T, S, U.T
    if len(S.shape) == 1:
        S = np.diag(S)
    V = Vt.T

    L = U.T @ C
    H = C - U @ L
    J, K = np.linalg.qr(H)

    flag_truncation = False
    null = np.zeros(shape=(K.shape[0], S.shape[1]))
    Q = np.block([
        [S, U.T @ C],
        [null, K]
    ])
    
    U_q, S_q, V_qt = np.linalg.svd(Q, full_matrices=False)
    V_q = V_qt.T

    if flag_truncation:
        U = U @ U_q[:r, :r]
    else:
        U = np.block([[U, J]]) @ U_q

        null_ur = np.zeros(shape=(V.shape[0], V_q.shape[0] - V.shape[1]))
        null_bl = np.zeros(shape=(C.shape[1], V.shape[1]))
        eye = np.eye(C.shape[1], V_q.shape[0] - V.shape[1])
        V = np.block([
            [V, null_ur],
            [null_bl, eye]
        ])
        V = V @ V_q

    U = U[:, :r]
    S = S_q[:r]
    Vt = V[:, :r].T

    U = reorthogonalization(U, tol)
    
    if update_rows:
        U, S, Vt = Vt.T, S, U.T
        
    return np.asarray(U), np.asarray(S), np.asarray(Vt)

def load_metrics(path, algorithm_name):
    pass