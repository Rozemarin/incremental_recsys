from src.splits import *
from src.utils import *
from src.algorithms import *
from src.visual import *
from src.predictions import *
from src.names import *

import time
import pandas as pd
from tqdm import tqdm


def a():
    return


def init_decomposition(data, alg_name, rank, seed, mode=FAST, A=None, At=None):

    if mode != FAST:
        A = get_matrix(data.training, None, None)
        At = A.T
    elif A is None or At is None:
        raise AttributeError('please provide A and At')
    
    oversampling = rank // 10

    start = time.time()
    if alg_name == SVD:
        U, S, Vt = sparse_svd(A, rank)
    elif alg_name == RAND_SVD:
        U, S, Vt = rand_svd(A, At, rank, power_iterations=0, oversampling=oversampling, seed=seed)
    elif alg_name == RAND_SVD_P1:
        U, S, Vt = rand_svd(A, At, rank, power_iterations=1, oversampling=oversampling, seed=seed)
    elif alg_name == RAND_SVD_P2:
        U, S, Vt = rand_svd(A, At, rank, power_iterations=2, oversampling=oversampling, seed=seed)
    elif alg_name == RAND_SVD_P3:
        U, S, Vt = rand_svd(A, At, rank, power_iterations=3, oversampling=oversampling, seed=seed)
    end = time.time()
    init_time = end - start

    return InitDecomposition(seed, U, S, Vt, init_time, data)
    

def metrics_experiment(
    algorithm_name,
    rank,
    K,
    update_type,
    init,
    seed=42,
    mode=TEST
):

    data = init.data
    results = ExperimentResult()

    if algorithm_name in [PSI1_ITEMS_FIRST, PSI1_ITEMS_FIRST_ORTH,
                            PSI2_ITEMS_FIRST, PSI2_ITEMS_FIRST_ORTH]:
        update_type = ITEMS_FIRST
    elif algorithm_name in [PSI1_USERS_FIRST, PSI1_USERS_FIRST_ORTH,
                            PSI2_USERS_FIRST, PSI2_USERS_FIRST_ORTH]:
        update_type = USERS_FIRST
    
    if 'ort' in algorithm_name:
        ort = True
    else:
        ort = False

    U, S, Vt = init.U, init.S, init.Vt
    total_users = len(data.training[USERID].unique())
    total_items = len(data.training[ITEMID].unique())

    if mode not in [FAST_VAL, VAL]:
        train_matrix = get_matrix(data.training, total_users, total_items)
        A = train_matrix
        At = A.T
        oversampling = rank // 10
        updated_training = data.training.copy()

    results.init_time = init.init_time

    total_metrics = np.array([[]]).reshape((N_METRICS, 0))
    initial_objs_metrics = np.array([[]]).reshape((N_METRICS, 0))

    new_total_users = total_users
    new_total_items = total_items

    unique_recomendations = {name : [] for name in METRICS_TYPES}

    prev_predictions = None
    prev_ids = None

    for i in tqdm(range(len(data.r_periods)), desc=algorithm_name):

        # relative error calculation
        if mode != FAST_VAL:
            if algorithm_name not in [ALL_RANDOM, MOST_POPULAR]:
                if len(S.shape) == 1:
                    Ar = (U * S) @ Vt
                else:
                    Ar = U @ S @ Vt
                rel_eps = np.linalg.norm(A.toarray() - Ar) / scipy.sparse.linalg.norm(A)
                if i == 0:
                    results.init_releps = rel_eps
                else:
                    results.dyn_releps.append(rel_eps)
            else:
                if i == 0:
                    results.init_releps = 1.0
                else:
                    results.dyn_releps.append(1.0)

        # n total users and items update
        total_users, total_items = new_total_users, new_total_items
        clean_next_portion, dirt = data.r_clean_next_portion[i], data.r_dirt[i]
        
        # calculating predictions
        start = time.time()
        if algorithm_name == ALL_RANDOM:
            # predictions = predict_random(data.r_A_act[i], updated_training, K)
            raise AttributeError('forget about ALL_RANDOM for now')
        elif algorithm_name == MOST_POPULAR:
            # predictions = predict_most_popular(data.r_A_act[i], updated_training, K)
            raise AttributeError('forget about MOST_POPULAR for now')
        else:
            predictions = predict(data.r_A_act[i], Vt, K, rank)
        end = time.time()
        # print('predictions calculation:', end - start)

        ### metrics
        cur_ids = data.r_users_ids[i][:data.r_n_real_active[i]]
        cur_predictions = predictions[:data.r_n_real_active[i]]

        calculated_metrics = np.array(
            calculate_metrics(
                                K, 
                                predictions, 
                                data.r_users_actual_watched[i], 
                                total_items,
                                cur_predictions,
                                cur_ids,
                                prev_predictions,
                                prev_ids
                            )).reshape(N_METRICS, -1)
        if i == 0:
            results.init_metrics = calculated_metrics
        else:
            total_metrics = np.append(total_metrics, calculated_metrics, axis=1)
            
        unique_recomendations['total_metrics'].append((total_items, get_unique_recomendations(predictions, K)))

        prev_ids = cur_ids
        prev_predictions = cur_predictions

        ### end metrics
        if mode == FAST_VAL or mode == VAL:
            break

        updated_training = pd.concat([updated_training, clean_next_portion, dirt]).sort_values(by=TIMESTAMP)
        new_total_users = len(updated_training[USERID].unique())
        new_total_items = len(updated_training[ITEMID].unique())

        diff_total_users = new_total_users - total_users
        diff_total_items = new_total_items - total_items


        if i == len(data.r_periods) - 1:
            break

        A = get_matrix(
            updated_training, 
            new_total_users, 
            new_total_items
        )
        At = A.T
            
        # Updating
        if algorithm_name in [PSI1_USERS_FIRST, PSI1_ITEMS_FIRST, 
                                PSI1_USERS_FIRST_ORTH, PSI1_ITEMS_FIRST_ORTH,
                                PSI2_USERS_FIRST, PSI2_ITEMS_FIRST,
                                PSI2_USERS_FIRST_ORTH, PSI2_ITEMS_FIRST_ORTH]:
            dA = get_matrix(clean_next_portion, total_users, total_items)
            dA12 = get_matrix(half_time_split(clean_next_portion), total_users, total_items)
            if not (diff_total_users == 0 and diff_total_items == 0):
                big_dA = get_matrix(dirt, new_total_users, new_total_items) # only dirty changes
            dAt = dA.T
            dA22 = dA - dA12

        start = time.time()

        if algorithm_name in [ALL_RANDOM, MOST_POPULAR]:
            pass
        elif algorithm_name == SVD:
            U, S, Vt = scipy.sparse.linalg.svds(A, k=rank) # sparse_svd(A, rank=rank)
        elif algorithm_name == RAND_SVD:
            U, S, Vt = rand_svd(A, At, rank, power_iterations=0, oversampling=oversampling, seed=seed)
        elif algorithm_name == RAND_SVD_P1:
            U, S, Vt = rand_svd(A, At, rank, power_iterations=1, oversampling=oversampling, seed=seed)
        elif algorithm_name == RAND_SVD_P2:
            U, S, Vt = rand_svd(A, At, rank, power_iterations=2, oversampling=oversampling, seed=seed)
        elif algorithm_name == RAND_SVD_P3:
            U, S, Vt = rand_svd(A, At, rank, power_iterations=3, oversampling=oversampling, seed=seed)

        elif algorithm_name in [REUSED_PROJECTOR_INCREMENTAL_V, 
                                # REUSED_PROJECTOR_INCREMENTAL_V_ORTH,
                                REUSED_PROJECTOR_INCREMENTAL_V_RSVD,
                                # REUSED_PROJECTOR_INCREMENTAL_V_RSVD_ORTH
                                ]:
            if diff_total_items > 0:
                big_dA = get_matrix(dirt, new_total_users, new_total_items) # only dirty changes 
                cold_items_hot_users_matrix = big_dA[:total_users, -diff_total_items:]
                U, S, Vt = matthew_brand(cold_items_hot_users_matrix, U, S, Vt, update_rows=False, reorthogonalization=ort)
            U, S, Vt = reused_projector_Vt(A, Vt, rank)
        elif algorithm_name in [REUSED_PROJECTOR_INCREMENTAL_U, 
                                # REUSED_PROJECTOR_INCREMENTAL_U_ORTH,
                                REUSED_PROJECTOR_INCREMENTAL_U_RSVD,
                                # REUSED_PROJECTOR_INCREMENTAL_U_RSVD_ORTH
                                ]:
            if diff_total_users > 0:
                big_dA = get_matrix(dirt, new_total_users, new_total_items) # only dirty changes 
                cold_users_hot_items_matrix = big_dA[-diff_total_users:, :total_items]
                U, S, Vt = matthew_brand(cold_users_hot_items_matrix, U, S, Vt, update_rows=True, reorthogonalization=ort)
            U, S, Vt = reused_projector_U(At, U, rank)
        elif algorithm_name in [PSI1_USERS_FIRST, PSI1_ITEMS_FIRST, 
                                PSI1_USERS_FIRST_ORTH, PSI1_ITEMS_FIRST_ORTH,
                                PSI2_USERS_FIRST, PSI2_ITEMS_FIRST,
                                PSI2_USERS_FIRST_ORTH, PSI2_ITEMS_FIRST_ORTH]:

            if algorithm_name in [PSI1_USERS_FIRST, PSI1_ITEMS_FIRST]:
                U, S, Vt = psi_step_1order(dA, dAt, U, S, Vt, r=rank)
            elif algorithm_name in [PSI2_USERS_FIRST, PSI2_ITEMS_FIRST]:
                U, S, Vt = psi_step_2order(dA, dAt, dA12, dA22, U, S, Vt, r=rank)

            if diff_total_users > 0 or diff_total_items > 0:
        
                if diff_total_users > 0 and diff_total_items == 0: # only new users
                    cold_users_hot_items_matrix = big_dA[-diff_total_users:, :total_items]
                    U, S, Vt = matthew_brand(cold_users_hot_items_matrix, U, S, Vt, r=rank, update_rows=True, reorthogonalization=ort)
                    
                elif diff_total_users == 0 and diff_total_items > 0: # only new items
                    cold_items_hot_users_matrix = big_dA[:total_users, -diff_total_items:]
                    U, S, Vt = matthew_brand(cold_items_hot_users_matrix, U, S, Vt, r=rank, update_rows=False, reorthogonalization=ort)
                    
                # elif pure_cold_exist: # only small svd is possible
                #     cold_ui_matrix = big_dA[-diff_total_users:, -diff_total_items:] # for svd
                #     U, S, Vt = pure_cold_svd_update(U, S, Vt, cold_ui_matrix, r=rank)

                elif diff_total_users > 0 and diff_total_items > 0:
                    
                    if update_type == USERS_FIRST:
                        cold_users_hot_items_matrix = big_dA[-diff_total_users:, :total_items]
                        U, S, Vt = matthew_brand(cold_users_hot_items_matrix, U, S, Vt, r=rank, update_rows=True, reorthogonalization=ort)
                        cold_items_all_users_matrix = big_dA[:, -diff_total_items:]
                        U, S, Vt = matthew_brand(cold_items_all_users_matrix, U, S, Vt, r=rank, update_rows=False, reorthogonalization=ort)

                    elif update_type == ITEMS_FIRST:
                        cold_items_hot_users_matrix = big_dA[:total_users, -diff_total_items:]
                        U, S, Vt = matthew_brand(cold_items_hot_users_matrix, U, S, Vt, r=rank, update_rows=False, reorthogonalization=ort)
                        cold_users_all_items_matrix = big_dA[-diff_total_users:, :]
                        U, S, Vt = matthew_brand(cold_users_all_items_matrix, U, S, Vt, r=rank, update_rows=True, reorthogonalization=ort)

                    elif update_type == COLD_FIRST:
                        raise NotImplementedError('pure_cold_first update not supported (yet)')
            
        else:
            raise NotImplementedError(f'no algorithm called "{algorithm_name}" supported')
        
        end = time.time()
        results.dyn_time.append(end - start)
    
    results.dyn_metrics = total_metrics
    results.initial_objs_metrics = initial_objs_metrics
    results.unique_recomendations = unique_recomendations
    return results
