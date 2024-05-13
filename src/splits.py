import numpy as np
from src.names import *
from src.utils import *
from tqdm import tqdm


def print_datasets_stats(df):
    print(f"# interactions: {df.shape[0]}")
    print(f"# users: {len(np.unique(df[USERID]))}")
    print(f"# items: {len(np.unique(df[ITEMID]))}")
    A = get_matrix(df, total_items=None, total_users=None)
    print(f"density: {(A.getnnz() / np.prod(A.shape)) * 100:.4f}%")


def remap_and_split(df, split_type, frac_train):
    # checkpoint = max(data['timestamp']) - dateutil.relativedelta.relativedelta(months=8)
    # training_, holdout_ = data[data['timestamp'] < checkpoint], data[data['timestamp'] >= checkpoint]
    checkpoint = int(len(df) * frac_train)
    training_, holdout_ = df[:checkpoint], df[checkpoint:]
    print(f"split type: {split_type}")
    print(f"checkpoing:", checkpoint)

    # чистим данные под вид сплита
    training, holdout = configure_split_type(training_, holdout_, type=split_type)
    assert training_.shape == training.shape
    clean_data = pd.concat([training, holdout])

    print(f"training.shape: {training.shape}")
    print(f"holdout.shape: {holdout.shape}")

    # делаем сквозные возрастающие айди для пользователей
    global_mapping_userid = update_mapping(clean_data, USERID)
    clean_data[USERID] = clean_data[USERID].map(global_mapping_userid)

    # делаем сквозные возрастающие айди для айтемов
    global_mapping_itemid = update_mapping(clean_data, ITEMID)
    clean_data[ITEMID] = clean_data[ITEMID].map(global_mapping_itemid) 

    print_datasets_stats(clean_data)

    # обратно делим на трейн и тест
    split_idx = training.shape[0]
    training, holdout = clean_data[:split_idx], clean_data[split_idx:]
    assert training_.shape == training.shape

    print(f"train %: {(training.shape[0]/(holdout.shape[0] + training.shape[0])) * 100}%")

    return training, holdout


def time_split(data, checkpoint, userid=USERID, movieid=ITEMID, timeid=TIMESTAMP):
    holdout = data[data[timeid] >= checkpoint]
    remaining = data[data[timeid] < checkpoint]
    return remaining, holdout, checkpoint


def interaction_split(data, n_interactions_in_period, userid=USERID, movieid=ITEMID, timeid=TIMESTAMP):
    assert len(data) >= n_interactions_in_period
    holdout = data[n_interactions_in_period:]
    remaining = data[:n_interactions_in_period]
    if holdout.shape[0] < n_interactions_in_period * 0.1:
        remaining = data
    return remaining, holdout, max(remaining[timeid])


# def active_users_split(data, n_active_users_in_period, userid=USERID, movieid=ITEMID, timeid=TIMESTAMP):
#     assert len(np.unique(data[userid])) > n_active_users_in_period
#     first_extra_user_id = np.unique(data[userid])[n_active_users_in_period]
#     split_index = (data[userid].values == first_extra_user_id).argmax()
#     holdout = data[split_index:]
#     remaining = data[:split_index]
#     return remaining, holdout, min(holdout[timeid])


def configure_split_type(train, test, type, userid=USERID, movieid=ITEMID, timeid=TIMESTAMP):
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
    
    split_date = min(holdout[TIMESTAMP])
    if periods_split_type == TIME_SPLIT:
        interval = (max(holdout[TIMESTAMP]) - split_date) / (n_periods + 1)
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

        total_users = len(updated_training[USERID].unique())
        total_movies = len(updated_training[ITEMID].unique())
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
            test_user_x_first_item = next_portion.groupby(USERID).head(1).sort_values(by=[USERID])
            users_ids = test_user_x_first_item[USERID]
            A_act = A[users_ids]
            users_actual_watched = test_user_x_first_item[ITEMID].to_numpy().reshape(-1, 1)

        #############
        r_users_ids.append(users_ids)
        r_A_act.append(A_act)
        r_users_actual_watched.append(users_actual_watched)
        r_n_real_active.append(len(np.unique(next_portion[USERID])))
        r_n_virtual_users.append(A_act.shape[0] - r_n_real_active[-1])
        #############

        if i == n_periods + 2:
            break

        if dirt.shape[0] == 0:
            r_n_pure_cold.append(0)
            continue

        n_pure_cold = len(dirt[
            (dirt[USERID] > total_users)
            & (dirt[ITEMID] > total_movies)
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

    data = ExperimentData(r_periods,
                            r_total_users_items,
                            r_users_ids,
                            r_A_act,
                            r_users_actual_watched,
                            r_clean_next_portion,
                            r_dirt,
                            r_n_virtual_users,
                            r_n_real_active,
                            r_n_pure_cold,
                            training,
                            holdout)

    return data