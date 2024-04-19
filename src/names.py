######
# algorithm names
SVD = 'svd'
PSI1 = 'psi 1 order'
PSI2 = 'psi 2 order'
RAND_SVD = 'rand svd'
REUSED_PROJECTOR_RANDOM = 'reused projector (random)'
REUSED_PROJECTOR_INCREMENTAL = 'reused projector (incremental)'
ALL_RANDOM = 'random'
MOST_POPULAR = 'popular'

ALL_ALG_NAMES = [SVD, 
                 PSI1, PSI2, 
                 RAND_SVD, 
                 REUSED_PROJECTOR_RANDOM, REUSED_PROJECTOR_INCREMENTAL, 
                 ALL_RANDOM, MOST_POPULAR]

######
# update types
USERS_FIRST = 'users first'
ITEMS_FIRST = 'items first'
COLD_FIRST = 'cold first'

######
# metrics names
N_METRICS = 4
METRICS_NAMES = [lambda k: f"hitrate@{k}", lambda k: f"MRR@{k}", 
                 lambda k: f"coverage@{k}", lambda k: f"stability@{k}"]
METRICS_NAMES_K = lambda k: [f"hitrate@{k}", f"MRR@{k}", f"coverage@{k}", f"stability@{k}"]
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