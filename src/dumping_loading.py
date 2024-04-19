from src.names import *
import json
import numpy as np


def dump_metrics(metrics_list, data_fname):
    json_dict = {}
    for alg_name, metrics in zip(ALL_ALG_NAMES, metrics_list):
        json_dict[alg_name] = {}
        for metric_type, metric in zip(METRICS_TYPES, metrics):
            json_dict[alg_name][metric_type] = metric.tolist()
        json_dict[alg_name]['unique_recs'] = {}
        for type, value in metrics[-1].items():
            json_dict[alg_name]['unique_recs'][type] = []
            for x, mlist in value:
                json_dict[alg_name]['unique_recs'][type].append([x, list(mlist)])


    def cast_type(container, from_types, to_types):
        if isinstance(container, dict):
            # cast all contents of dictionary 
            return {cast_type(k, from_types, to_types): cast_type(v, from_types, to_types) for k, v in container.items()}
        elif isinstance(container, list):
            # cast all contents of list 
            return [cast_type(item, from_types, to_types) for item in container]
        else:
            for f, t in zip(from_types, to_types):
                # if item is of a type mentioned in from_types,
                # cast it to the corresponding to_types class
                if isinstance(container, f):
                    return t(container)
            # None of the above, return without casting 
            return container

    with open(f'metrics/{data_fname}', 'wt') as f:
        json.dump(cast_type(json_dict, [np.int64, np.float64], [int, float]), f, indent=4)