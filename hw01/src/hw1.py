#!/usr/bin/env python

import numpy as np
import math
def split_into_train_and_test(x_all_LF, frac_test=0.5, random_state=None):
    tmp = x_all_LF.copy()
    train_len = math.floor(len(tmp) * (1-frac_test))
    if random_state is None:
        np.random.shuffle(tmp)
    elif isinstance(random_state,int):
        np.random.RandomState(random_state).shuffle(tmp)
    else:
        random_state.shuffle(tmp)
    return tmp[:train_len], tmp[train_len:]

def calc_k_nearest_neighbors(data_NF, query_QF, K=1):
    l = []
    for v2 in query_QF:
        d = {}
        tmp = []
        for v1 in data_NF:
            dist = distance(v1,v2)
            if dist in d:
                d[dist].append(v1)
            else:
                d[dist] = [v1]
        for i in range(K):
            cur = min(d.keys())
            tmp.append(d[cur].pop())
            if len(d[cur]) == 0:
                d.pop(cur)
        l.append(tmp)
    return np.array(l)

def distance(v1,v2):
    sum = 0
    for i in range(len(v1)):
        sum += (v1[i] - v2[i]) * (v1[i] - v2[i])
    return math.sqrt(sum)