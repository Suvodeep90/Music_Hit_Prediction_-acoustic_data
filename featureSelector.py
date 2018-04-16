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
from sklearn.feature_selection import SelectKBest,RFECV
from sklearn.model_selection import StratifiedKFold


class featureSelection():
    
    def __init__(self, data, clf):
        self.feature_clf = None
        self.data = data
        self.class_label = self.data.columns[-1]
        self.columns = self.data.columns[:-1]
        self.feature_subset = None
        self.selected_feature = None
        self.train_X = None
        self.train_y = None
        self.clf = clf
        self.set_data()
        
    
    def set_data(self):
        self.train_y = self.data[self.class_label]
        self.train_X = self.data[self.columns]
        

    def score(self):
        data_subset = self.data[self.feature_subset].join(self.data[self.class_label])
        uniques_data_points = data_subset.drop_duplicates()    
        subsum = 0
        for i in range(uniques_data_points.shape[0] - 1):
            row = uniques_data_points.iloc[i]
            matches = data_subset[data_subset == row].dropna()
            if matches.shape[0] <= 1:
                continue
            D = matches.shape[0]
            label = matches[self.class_label] == float(matches.mode()[self.class_label])
            M = matches[label].shape[0]
            subsum += (D - M)
        return 1 - subsum / data_subset.shape[0]

    def consistency_subset(self):
        starts_time = time.time()
        last_improve_at = time.time()
        best = [0, None]
        while time.time() - last_improve_at < 5 or time.time() - starts_time < 10:
            rand_selected_attr = [random.choice([0, 1]) for _ in range(len(self.columns))]
            if not sum(rand_selected_attr):
                continue
            self.feature_subset = [self.columns[i] for i, v in enumerate(rand_selected_attr) if v]
            score = self.score()
            if score > best[0]:
                best = [score, self.feature_subset]
                last_improve_at = time.time()
        
        self.selected_feature = best[1] + [self.class_label]
        return self.data[self.selected_feature]
    
    def selectkBest(self):
        self.feature_clf = SelectKBest()
        self.feature_clf.fit(self.train_X,self.train_y)
        self.selected_feature = self.columns[self.feature_clf.get_support(indices=True)]
        self.feature_subset = self.data[self.selected_feature]
        self.train_y = pd.DataFrame(self.train_y)
        self.feature_subset = self.feature_subset.join(self.train_y)
        return self.feature_subset
    
    def recursive_feature_selector(self):
        self.feature_clf = RFECV(estimator=self.clf, step=1, cv=StratifiedKFold(2),
              scoring='f1_weighted')
        self.feature_clf.fit(self.train_X,self.train_y)
        self.selected_feature = self.columns[self.feature_clf.get_support(indices=True)]
        self.feature_subset = self.data[self.selected_feature]
        self.train_y = pd.DataFrame(self.train_y)
        self.feature_subset = self.feature_subset.join(self.train_y)
        return self.feature_subset
