import numpy as np
from src.names import *
from src.utils import *
from tqdm import tqdm

def time_split(data, checkpoint, userid='userid', movieid='movieid', timeid='timestamp'):
    holdout = data[data[timeid] >= checkpoint]
    remaining = data[data[timeid] < checkpoint]
    return remaining, holdout, checkpoint


def interaction_split(data, n_interactions_in_period, userid='userid', movieid='movieid', timeid='timestamp'):
    assert len(data) >= n_interactions_in_period
    holdout = data[n_interactions_in_period:]
    remaining = data[:n_interactions_in_period]
    if holdout.shape[0] < n_interactions_in_period * 0.1:
        remaining = data
    return remaining, holdout, max(remaining[timeid])


# def active_users_split(data, n_active_users_in_period, userid='userid', movieid='movieid', timeid='timestamp'):
#     assert len(np.unique(data[userid])) > n_active_users_in_period
#     first_extra_user_id = np.unique(data[userid])[n_active_users_in_period]
#     split_index = (data[userid].values == first_extra_user_id).argmax()
#     holdout = data[split_index:]
#     remaining = data[:split_index]
#     return remaining, holdout, min(holdout[timeid])


def configure_split_type(train, test, type, userid='userid', movieid='movieid', timeid='timestamp'):
    """
    type 0: hot users, hot items
    type 1: hot users + cold users, hot items
    type 2: hot users, hot items + cold items
    type 3: hot users + cold_users, hot items + cold items
    """
    if type == 0:
        test = test[test[userid].isin(train[userid]) 
                    & test[movieid].isin(train[movieid])]
    elif type == 1:
        test = test[test[movieid].isin(train[movieid])]
    elif type == 2:
        test = test[test[userid].isin(train[userid])]
    elif type == 3:
        pass
    else:
        raise AttributeError
    return train, test


def prepare_split(training, holdout, n_periods, flag_virtual, periods_split_type, n_active_users_in_period=None):
    
    split_date = min(holdout['timestamp'])
    if periods_split_type == TIME_SPLIT:
        interval = (max(holdout['timestamp']) - split_date) / (n_periods + 1)
        periods = [split_date + i * interval for i in range(1, n_periods + 2)]
    elif periods_split_type == ACTIVE_USERS_SPLIT:
        assert n_active_users_in_period is not None
        periods = []
    elif periods_split_type == INTERACTION_SPLIT:
        n_interactions_in_period = len(holdout) // (n_periods + 1)
        periods = []
    else:
        raise AttributeError(f'no split type called "{periods_split_type}" supported')
    periods = [split_date] + periods

    updated_training = training.copy()
    saved_holdout = holdout.copy()

    r_periods = []
    r_total_users_items = []

    r_users_ids = []
    r_A_act = []
    r_users_actual_watched = []
    r_clean_next_portion = []
    r_dirt = []

    r_n_virtual_users = []
    r_n_real_active = []
    r_n_pure_cold = []
    for i in tqdm(range(1, n_periods + 2)):

        total_users = len(updated_training['userid'].unique())
        total_movies = len(updated_training['movieid'].unique())
        ######
        r_total_users_items.append((total_users, total_movies))
        ######
        A = get_matrix(updated_training, total_users, total_movies)
    
        if periods_split_type == TIME_SPLIT:
            dirty_next_portion, saved_holdout, checkpoint = time_split(saved_holdout, periods[i])
        elif periods_split_type == ACTIVE_USERS_SPLIT:
            # dirty_next_portion, saved_holdout, checkpoint = active_users_split(saved_holdout, n_active_users_in_period)
            # periods.append(checkpoint)
            raise AttributeError(f'split type {periods_split_type} not supported')
        elif periods_split_type == INTERACTION_SPLIT:
            dirty_next_portion, saved_holdout, checkpoint = interaction_split(saved_holdout, n_interactions_in_period)
            periods.append(checkpoint)
        else:
            raise AttributeError(f'split type {periods_split_type} not supported')

        next_portion, dirt = collect_dirt(dirty_next_portion, total_users, total_movies)
        #############
        r_periods.append(checkpoint)
        r_clean_next_portion.append(next_portion)
        r_dirt.append(dirt)
        #############

        if flag_virtual:
            A_act, users_actual_watched, users_ids = create_virtual_users(A, next_portion)
        else:
            test_user_x_first_item = next_portion.groupby('userid').head(1).sort_values(by=['userid'])
            users_ids = test_user_x_first_item['userid']
            A_act = A[users_ids]
            users_actual_watched = test_user_x_first_item['movieid'].to_numpy().reshape(-1, 1)

        #############
        r_users_ids.append(users_ids)
        r_A_act.append(A_act)
        r_users_actual_watched.append(users_actual_watched)
        r_n_real_active.append(len(np.unique(next_portion['userid'])))
        r_n_virtual_users.append(A_act.shape[0] - r_n_real_active[-1])
        #############

        if i == n_periods + 2:
            break

        if dirt.shape[0] == 0:
            r_n_pure_cold.append(0)
            continue

        n_pure_cold = len(dirt[
            (dirt['userid'] > total_users)
            & (dirt['movieid'] > total_movies)
        ])
        #############
        r_n_pure_cold.append(n_pure_cold)
        #############
        updated_training = pd.concat([updated_training, dirty_next_portion])

    ###### tests ######
    ### dirt + clean = holdout
    x = sum([r_clean_next_portion[i].shape[0] + r_dirt[i].shape[0]
                for i in range(len(r_dirt))])
    y = holdout.shape[0]
    assert x == y

    return (
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
    )