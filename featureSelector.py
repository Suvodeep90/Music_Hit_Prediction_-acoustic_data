#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 11:41:32 2018

@author: suvodeep
"""
from __future__ import division
import pandas as pd
import math
import numpy as np
import scipy as sc
import random
import pdb
import time


def _ent(data):
    p_data = data.value_counts() / len(data)  # calculates the probabilities
    entropy = sc.stats.entropy(p_data)  # input probabilities to get the entropy
    return entropy


def gain_rank(df):
    H_C = _ent(df.iloc[:, -1])
    weights = pd.DataFrame(data=np.zeros([1, df.shape[1] - 1]), columns=df.columns[:-1])

    types_C = set(df.iloc[:, -1])
    target = df.columns[-1]
    for a_i, a in enumerate(df.columns[:-1]):  # for each attribute a
        for typea in set(df.loc[:, a]):  # each class of attribute a
            selected_a = df[df[a] == typea]
            sub = 0
            for typec in types_C:
                p_c_a = selected_a[selected_a[target] == typec].shape[0] / selected_a.shape[0]
                if p_c_a == 0:
                    continue
                sub += p_c_a * math.log(p_c_a, 2)
            weights.loc[0, a] += -1 * selected_a.shape[0] / df.shape[0] * sub

    weights = H_C - weights
    weights[df.columns[-1]] = 1
    weights = weights.append([weights] * (df.shape[0] - 1), ignore_index=False)
    weights.index = df.index

    res = weights * df
    return res




def consistency_subset(df):

    def consistency(sdf, classes):
        sdf = sdf.join(classes)
        uniques = sdf.drop_duplicates()
        target = classes.name

        subsum = 0

        for i in range(uniques.shape[0] - 1):
            row = uniques.iloc[i]
            matches = sdf[sdf == row].dropna()
            if matches.shape[0] <= 1: continue
            D = matches.shape[0]
            M = matches[matches[target] == float(matches.mode()[target])].shape[0]
            subsum += (D - M)

        return 1 - subsum / sdf.shape[0]

    features = df.columns[:-1]
    target = df.columns[-1]

    hc_starts_at = time.time()
    lst_improve_at = time.time()
    best = [0, None]
    while time.time() - lst_improve_at < 1 or time.time() - hc_starts_at < 5:
        # during of random_config search -> at most 5 seconds. if no improve by 1 second, then stop
        selects = [random.choice([0, 1]) for _ in range(len(features))]
        if not sum(selects): continue
        fs = [features[i] for i, v in enumerate(selects) if v]
        score = consistency(df[fs], df[target])
        if score > best[0]:
            best = [score, fs]
            lst_improve_at = time.time()

    selected_features = best[1] + [target]
    return df[selected_features]


