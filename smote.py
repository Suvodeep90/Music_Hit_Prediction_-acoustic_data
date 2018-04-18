# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 20:07:02 2018

@author: suvod
"""

from __future__ import print_function, division
import random
from collections import Counter
import pandas as pd
from sklearn.neighbors import NearestNeighbors,KNeighborsClassifier
from sklearn import tree


class smote(object):
    def __init__(self, pd_data, k_neighbor=10,m_neighbor = 10):
        self.set_data(pd_data)
        self.k_neighbor = k_neighbor
        self.m_neighbor = m_neighbor
        self.label_num = len(set(pd_data[pd_data.columns[-1]].values))
        self.train_X = None
        self.train_X_all = None
        self.train_y = None
        self.clf = None
        self.clf_quality = None
    
    def set_data(self, pd):
        if not pd.empty:
          self.data = pd
        else:
          raise ValueError(
            "The last column of pd_data should be string as class label")
          
    def get_majority_num(self):
        lCount = Counter(self.data[self.data.columns[-1]].values)
        majority_num = max(lCount.values())
        return lCount,majority_num

    def get_neighbors(self,data_no_label):
      rand_sample_idx = random.randint(0, len(data_no_label) - 1)
      rand_sample = data_no_label[rand_sample_idx]
      distance, ngbr = self.clf.kneighbors(rand_sample.reshape(1, -1))
      rand_ngbr_idx = random.randint(0, len(ngbr))
      return data_no_label[rand_ngbr_idx], rand_sample
  
    def get_data(self,label):
        self.train_y = self.data[self.data.columns[-1]]
        self.train_X_all = self.data[self.data.columns[:-1]].values
        df = self.data.loc[self.train_y == label]
        self.train_X = df[self.data.columns[:-1]].values
        print(self.train_X_all.shape,self.train_y.shape)

    def fit_transform(self):
        df = self.data.values.tolist()
        classCount, majority_num = self.get_majority_num()
        for label, num in classCount.items():
            if num < majority_num:
                minority_number = majority_num - num
                self.get_data(label)
                if len(self.train_X) < self.k_neighbor:
                    self.neighbor = len(self.train_X)
                self.clf = NearestNeighbors(n_neighbors=self.k_neighbor).fit(self.train_X)
                self.clf_quality = KNeighborsClassifier(n_neighbors=self.m_neighbor).fit(self.train_X_all, self.train_y)
                count = 0
                while(count<minority_number):
                    neighbor, sample = self.get_neighbors(self.train_X)
                    new_row = []
                    new_data_point = []
                    for i, one in enumerate(neighbor): 
                        gap = random.random()
                        new_data_point.append(max(0, sample[i] + (sample[i] - one) * gap))
                    predicted_label = self.clf_quality.predict(new_data_point)                  
                    if predicted_label == label:
                        new_row = new_data_point
                        new_row.append(label)
                        df.append(new_row)
                        count += 1                    
        return pd.DataFrame(df)
