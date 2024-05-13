import matplotlib.pyplot as plt
from dataclasses import dataclass
import numpy as np
import matplotlib.colors


class ExperimentData():
    def __init__(self,
                    r_periods,
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
                    holdout
        ):
        self.r_periods = r_periods
        self.r_total_users_items = r_total_users_items
        self.r_users_ids = r_users_ids
        self.r_A_act = r_A_act
        self.r_users_actual_watched = r_users_actual_watched
        self.r_clean_next_portion = r_clean_next_portion
        self.r_dirt = r_dirt
        self.r_n_virtual_users = r_n_virtual_users
        self.r_n_real_active = r_n_real_active
        self.r_n_pure_cold = r_n_pure_cold
        self.training = training
        self.holdout = holdout


class InitDecomposition():
    def __init__(self, id, U, S, Vt, init_time, data):
        self.id = id
        self.U = U
        self.S = S
        self.Vt = Vt
        self.init_time = init_time
        self.data = data
    

class ExperimentResult():
    def __init__(self):
        self.init_metrics: float = -1
        self.dyn_metrics: list = []
        self.init_time: float = -1
        self.dyn_time: list = []
        self.init_releps: float = -1
        self.dyn_releps: list = []

        self.initial_objs_metrics: list = []
        self.unique_recomendations: dict = {}


TEST = 0
VAL = 1
FAST_VAL = 2
FAST = 3

######
# columns names
USERID = 'userid'
ITEMID = 'itemid'
RATING = 'rating'
TIMESTAMP = 'timestamp'
COLUMNS_NAMES = [USERID, ITEMID, RATING, TIMESTAMP]

######
# algorithm names and colors

def categorical_cmap(nc, nsc, cmap="tab10", continuous=False):
    if nc > plt.get_cmap(cmap).N:
        raise ValueError("Too many categories for colormap.")
    if continuous:
        ccolors = plt.get_cmap(cmap)(np.linspace(0,1,nc))
    else:
        ccolors = plt.get_cmap(cmap)(np.arange(nc, dtype=int))
    cols = np.zeros((nc*nsc, 3))
    for i, c in enumerate(ccolors):
        chsv = matplotlib.colors.rgb_to_hsv(c[:3])
        arhsv = np.tile(chsv,nsc).reshape(nsc,3)
        arhsv[:,1] = np.linspace(chsv[1],0.25,nsc)
        arhsv[:,2] = np.linspace(chsv[2],1,nsc)
        rgb = matplotlib.colors.hsv_to_rgb(arhsv)
        cols[i*nsc:(i+1)*nsc,:] = rgb       
    cmap = matplotlib.colors.ListedColormap(cols)
    return cmap

base_colors = 10
nvariations = 4
cmap = categorical_cmap(base_colors, nvariations, cmap="tab10")
# cmap = plt.get_cmap("tab20")

SVD = 'svd'

PSI1 = 'psi 1'
PSI1_USERS_FIRST = 'psi 1 users fist'
PSI1_USERS_FIRST_ORTH = 'psi 1 users fist ort'
PSI1_ITEMS_FIRST = 'psi 1 items fist'
PSI1_ITEMS_FIRST_ORTH = 'psi 1 items fist ort'

PSI2 = 'psi 2'
PSI2_USERS_FIRST = 'psi 2 users fist'
PSI2_USERS_FIRST_ORTH = 'psi 2 users fist ort'
PSI2_ITEMS_FIRST = 'psi 2 items fist'
PSI2_ITEMS_FIRST_ORTH = 'psi 2 items fist ort'
# PSI1 = 'psi 1 order'
# PSI2 = 'psi 2 order'
RAND_SVD = 'rand svd'
RAND_SVD_P1 = 'rand svd pi=1'
RAND_SVD_P2 = 'rand svd pi=2'
RAND_SVD_P3 = 'rand svd pi=3'

REUSED_PROJECTOR_RANDOM = 'reused projector (random)'

REUSED_PROJECTOR_INCREMENTAL_V = 'reused projector V (svd)'
# REUSED_PROJECTOR_INCREMENTAL_V_ORTH = 'reused projector V ort (svd)'
REUSED_PROJECTOR_INCREMENTAL_U = 'reused projector U (svd)'
# REUSED_PROJECTOR_INCREMENTAL_U_ORTH = 'reused projector U ort (svd)'

REUSED_PROJECTOR_INCREMENTAL_V_RSVD = 'reused projector V (rsvd)'
REUSED_PROJECTOR_INCREMENTAL_U_RSVD = 'reused projector U (rsvd)'
# REUSED_PROJECTOR_INCREMENTAL_V_RSVD_ORTH = 'reused projector V ort (rsvd)'
# REUSED_PROJECTOR_INCREMENTAL_U_RSVD_ORTH = 'reused projector U ort (rsvd)'

ALL_RANDOM = 'random'
MOST_POPULAR = 'popular'

ALL_ALG_COLORS = {
    SVD: cmap(0),
    PSI1: cmap(4),

    PSI1_USERS_FIRST: cmap(4),
    PSI1_ITEMS_FIRST: cmap(6),
    PSI1_USERS_FIRST_ORTH: cmap(5),
    PSI1_ITEMS_FIRST_ORTH: cmap(7),

    PSI2: cmap(8),

    PSI2_USERS_FIRST: cmap(8),
    PSI2_ITEMS_FIRST: cmap(10),
    PSI2_USERS_FIRST_ORTH: cmap(9),
    PSI2_ITEMS_FIRST_ORTH: cmap(11),

    RAND_SVD: cmap(12),
    RAND_SVD_P1: cmap(13),
    RAND_SVD_P2: cmap(14),
    RAND_SVD_P3: cmap(15),

    REUSED_PROJECTOR_INCREMENTAL_V: cmap(16),
    # REUSED_PROJECTOR_INCREMENTAL_V_ORTH: cmap(17),
    REUSED_PROJECTOR_INCREMENTAL_V_RSVD: cmap(18),
    # REUSED_PROJECTOR_INCREMENTAL_V_RSVD_ORTH: cmap(19),

    REUSED_PROJECTOR_INCREMENTAL_U: cmap(24),
    # REUSED_PROJECTOR_INCREMENTAL_U_ORTH: cmap(25),
    REUSED_PROJECTOR_INCREMENTAL_U_RSVD: cmap(26),
    # REUSED_PROJECTOR_INCREMENTAL_U_RSVD_ORTH: cmap(27),
    
    ALL_RANDOM: cmap(34),
    MOST_POPULAR: cmap(39)
}

ALL_ALG_NAMES = [SVD, 
                 PSI1_USERS_FIRST, 
                 PSI1_USERS_FIRST_ORTH,
                 PSI1_ITEMS_FIRST,
                 PSI1_ITEMS_FIRST_ORTH,
                 PSI2_USERS_FIRST, 
                 PSI2_USERS_FIRST_ORTH,
                 PSI2_ITEMS_FIRST,
                 PSI2_ITEMS_FIRST_ORTH,
                 RAND_SVD,
                 RAND_SVD_P1,
                 RAND_SVD_P2,
                 RAND_SVD_P3,
                 # REUSED_PROJECTOR_RANDOM, 
                 REUSED_PROJECTOR_INCREMENTAL_V,
                 # REUSED_PROJECTOR_INCREMENTAL_V_ORTH,
                 REUSED_PROJECTOR_INCREMENTAL_V_RSVD,
                 # REUSED_PROJECTOR_INCREMENTAL_V_RSVD_ORTH,

                 REUSED_PROJECTOR_INCREMENTAL_U,
                 # REUSED_PROJECTOR_INCREMENTAL_U_ORTH,
                 REUSED_PROJECTOR_INCREMENTAL_U_RSVD,
                 # REUSED_PROJECTOR_INCREMENTAL_U_RSVD_ORTH,

                 ALL_RANDOM, MOST_POPULAR
                 ]

DET_ALG_GROUP = [SVD, 
                 PSI1_USERS_FIRST, 
                 PSI1_USERS_FIRST_ORTH,
                 PSI1_ITEMS_FIRST,
                 PSI1_ITEMS_FIRST_ORTH,
                 PSI2_USERS_FIRST, 
                 PSI2_USERS_FIRST_ORTH,
                 PSI2_ITEMS_FIRST,
                 PSI2_ITEMS_FIRST_ORTH,
                 REUSED_PROJECTOR_INCREMENTAL_V,
                 # REUSED_PROJECTOR_INCREMENTAL_V_ORTH,
                 REUSED_PROJECTOR_INCREMENTAL_U,
                 # REUSED_PROJECTOR_INCREMENTAL_U_ORTH
                 ]

RAND_ALG_GROUP = [ 
                 RAND_SVD,
                 RAND_SVD_P1,
                 RAND_SVD_P2,
                 RAND_SVD_P3,
                 REUSED_PROJECTOR_INCREMENTAL_V_RSVD,
                 # REUSED_PROJECTOR_INCREMENTAL_V_RSVD_ORTH,
                 REUSED_PROJECTOR_INCREMENTAL_U_RSVD,
                 # REUSED_PROJECTOR_INCREMENTAL_U_RSVD_ORTH
                 ]


######
# update types
USERS_FIRST = 'users first'
ITEMS_FIRST = 'items first'
COLD_FIRST = 'cold first'

######
# metrics names
N_METRICS = 5
METRICS_NAMES = [lambda k: f"hitrate@{k}", lambda k: f"MRR@{k}", lambda k: f"NDCG@{k}",
                 lambda k: f"coverage@{k}", lambda k: f"stability@{k}"]
METRICS_NAMES_K = lambda k: [f"hitrate@{k}", f"MRR@{k}", f"NDCG@{k}", 
                             f"coverage@{k}", f"stability@{k}"]
METRICS_TYPES = ['total_metrics', 'initial_objs_metrics', 'new_objs_metrics']

######
# split types
H_USERS_H_ITEMS = 0
CH_USERS_H_ITEMS = 1
H_USERS_CH_ITEMS = 2
CH_USERS_CH_ITEMS = 3

TIME_SPLIT = 'time'
INTERACTION_SPLIT = 'interactions'
ACTIVE_USERS_SPLIT = 'active users'

# gowalla yelp kindle store behance beer