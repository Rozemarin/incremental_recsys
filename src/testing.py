from src.splits import *
from src.utils import *
from src.algorithms import *
from src.visual import *
from src.names import *
from src.predictions import *

import time
import pandas as pd
from tqdm import tqdm


def metrics_experiment(
    algorithm_name,
    rank,
    K,
    update_type,
    training,
    data
):
    (
        r_periods,
        r_total_users_items,
        r_users_ids,
        r_A_act,
        r_users_actual_watched,
        r_clean_next_portion,
        r_dirt,
        r_n_virtual_users,
        r_n_real_active,
        r_n_pure_cold
    ) = data

    total_users = len(training['userid'].unique())
    total_movies = len(training['movieid'].unique())

    train_matrix = get_matrix(training, total_users, total_movies)
    A = train_matrix
    U, S, Vt = sparse_svd(train_matrix, rank)
    dA = None

    updated_training = training.copy()

    total_metrics = np.array([[]]).reshape((N_METRICS, 0))
    initial_objs_metrics = np.array([[]]).reshape((N_METRICS, 0))
    new_objs_metrics = np.array([[]]).reshape((N_METRICS, 0))

    new_total_users = total_users
    new_total_movies = total_movies

    unique_recomendations = {name : [] for name in METRICS_TYPES}

    prev_predictions = None
    prev_ids = None

    for i in tqdm(range(len(r_periods)), desc=algorithm_name):

        total_users, total_movies = new_total_users, new_total_movies
        clean_next_portion, dirt = r_clean_next_portion[i], r_dirt[i]
        
        if algorithm_name == ALL_RANDOM:
            predictions = predict_random(r_A_act[i], updated_training, K)
        elif algorithm_name == MOST_POPULAR:
            predictions = predict_most_popular(r_A_act[i], updated_training, K)
        else:
            predictions = predict(r_A_act[i], Vt, K)

        ### metrics
        cur_ids = r_users_ids[i][:r_n_real_active[i]]
        cur_predictions = predictions[:r_n_real_active[i]]

        total_metrics = np.append(
            total_metrics, 
            np.array(
                calculate_metrics(
                    K, 
                    predictions, 
                    r_users_actual_watched[i], 
                    total_movies,
                    cur_predictions,
                    cur_ids,
                    prev_predictions,
                    prev_ids
                )
            ).reshape(N_METRICS, -1), axis=1)
        unique_recomendations['total_metrics'].append((total_movies, get_unique_recomendations(predictions, K)))

        prev_ids = cur_ids
        prev_predictions = cur_predictions
        ### end metrics

        updated_training = pd.concat([updated_training, clean_next_portion, dirt]).sort_values(by='timestamp')
        new_total_users = len(updated_training['userid'].unique())
        new_total_movies = len(updated_training['movieid'].unique())

        diff_total_users = new_total_users - total_users
        diff_total_movies = new_total_movies - total_movies

        if i == len(r_periods):
            break

        A = get_matrix(
            updated_training, 
            new_total_users, 
            new_total_movies
        )
            
        # Updating
        if algorithm_name in [ALL_RANDOM, MOST_POPULAR]:
            continue
        if algorithm_name == SVD:
            U, S, Vt = sparse_svd(A, rank=rank)
            continue
        elif algorithm_name == RAND_SVD:
            U, S, Vt = rand_svd(A, rank)
            continue
        elif algorithm_name == PSI1:
            dA = get_matrix(clean_next_portion, total_users, total_movies)
            U, S, Vt = psi_step_1order(dA, U, S, Vt, r=rank)
        elif algorithm_name == PSI2:
            dA = get_matrix(clean_next_portion, total_users, total_movies)
            dA12 = get_matrix(half_time_split(clean_next_portion), total_users, total_movies)
            U, S, Vt = psi_step_2order(dA, dA12, U, S, Vt, r=rank)
        elif algorithm_name == REUSED_PROJECTOR_RANDOM:
            if diff_total_users > 0:
                U = add_random_embeddings(U, diff_total_users)
            U, S, Vt = reused_projector_svd(A, U, rank)
            continue
        elif algorithm_name == REUSED_PROJECTOR_INCREMENTAL:
            if diff_total_users > 0:
                big_dA = get_matrix(dirt, new_total_users, new_total_movies) # only dirty changes 
                cold_users_hot_items_matrix = big_dA[-diff_total_users:, :total_movies]
                U, S, Vt = matthew_brand(cold_users_hot_items_matrix, U, S, Vt, r=rank, update_rows=True)
            U, S, Vt = reused_projector_svd(A, U, rank)
            continue
        else:
            raise NotImplementedError(f'no algorithm called "{algorithm_name}" supported')
        
        # here we have only PSI-1 and PSI-2
        U, S, Vt = U[:, :rank], S[:rank], Vt[:rank, :]
        if diff_total_users == 0 and diff_total_movies == 0:
            continue
        
        pure_cold_exist = len(dirt[
            (dirt['userid'] > total_users)
            & (dirt['movieid'] > total_movies)
        ]) > 0
        assert (pure_cold_exist and (r_n_pure_cold[i] > 0)) or (not pure_cold_exist and (r_n_pure_cold[i] == 0))

        big_dA = get_matrix(dirt, new_total_users, new_total_movies) # only dirty changes

        if diff_total_users > 0 and diff_total_movies == 0: # only new users
            cold_users_hot_items_matrix = big_dA[-diff_total_users:, :total_movies]
            U, S, Vt = matthew_brand(cold_users_hot_items_matrix, U, S, Vt, r=rank, update_rows=True)
            
        elif diff_total_users == 0 and diff_total_movies > 0: # only new items
            cold_items_hot_users_matrix = big_dA[:total_users, -diff_total_movies:]
            U, S, Vt = matthew_brand(cold_items_hot_users_matrix, U, S, Vt, r=rank, update_rows=False)
            
        elif diff_total_users > 0 and diff_total_movies > 0:
            if update_type == USERS_FIRST:
                cold_users_hot_items_matrix = big_dA[-diff_total_users:, :total_movies]
                U, S, Vt = matthew_brand(cold_users_hot_items_matrix, U, S, Vt, r=rank, update_rows=True)
                cold_items_all_users_matrix = big_dA[:, -diff_total_movies:]
                U, S, Vt = matthew_brand(cold_items_all_users_matrix, U, S, Vt, r=rank, update_rows=False)

            elif update_type == ITEMS_FIRST:
                cold_items_hot_users_matrix = big_dA[:total_users, -diff_total_movies:]
                U, S, Vt = matthew_brand(cold_items_hot_users_matrix, U, S, Vt, r=rank, update_rows=False)
                cold_users_all_items_matrix = big_dA[-diff_total_users:, :]
                U, S, Vt = matthew_brand(cold_users_all_items_matrix, U, S, Vt, r=rank, update_rows=True)

            elif update_type == COLD_FIRST:
                raise NotImplementedError('pure_cold_first update not supported (yet)')
            
        # elif pure_cold_exist: # only small svd is possible
        #     cold_ui_matrix = big_dA[-diff_total_users:, -diff_total_movies:] # for svd
        #     U, S, Vt = pure_cold_svd_update(U, S, Vt, cold_ui_matrix, r=rank)
        

    return total_metrics, initial_objs_metrics, new_objs_metrics, unique_recomendations
