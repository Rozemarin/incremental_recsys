import numpy as np


def remove_history(A_act, predictions, K):
    new_predictions = np.empty((predictions.shape[0], K))
    for i, act_ in enumerate(A_act):
        act = act_.toarray()[0]
        nonzero = act.nonzero()[0]
        mask = ~np.isin(predictions[i], nonzero)
        new_predictions[i] = predictions[i][mask][:K]
    return new_predictions


def predict(A_act, Vt, K):
    # r^T = a^T V V^T   (:) = (..)(mxn)(nxm)
    # r = V V^T a     (..) = (mxn)(nxm)(:)
    # A_act are lines, while a is column
    # R = (Vt.T @ Vt @ A_act.T).T
    R = A_act @ Vt.T @ Vt
    predictions = np.argsort(-R)
    assert predictions.shape[0] == A_act.shape[0]
    return remove_history(A_act, predictions, K)


def predict_most_popular(A_act, training, K):
    k_popular = training.groupby('movieid', as_index=False).count()['movieid'].to_numpy()
    predictions = np.tile(k_popular, (A_act.shape[0], 1))
    return remove_history(A_act, predictions, K)


def predict_random(A_act, training, K):
    movieids = np.unique(training['movieid'])
    predictions = np.random.choice(movieids, size=(A_act.shape[0], A_act.shape[1] + K*2))
    return remove_history(A_act, predictions, K)
