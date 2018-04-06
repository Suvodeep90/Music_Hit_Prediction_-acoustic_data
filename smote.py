# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 20:07:02 2018

@author: suvod
"""

from __future__ import print_function, division
import unittest
import random
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors as NN


class smote(object):
  def __init__(self, pd_data, neighbor=5, up_to_num=[]):
    self.set_data(pd_data)
    self.neighbor = neighbor
    self.up_to_max = False
    self.up_to_num = up_to_num
    self.label_num = len(set(pd_data[pd_data.columns[-1]].values))
    if up_to_num:
      label_num = len(set(pd_data[pd_data.columns[-1]].values))
      if label_num - 1 != len(up_to_num):
        raise ValueError(
          "should set smoted size for " + str(label_num - 1) + " minorities")
      self.up_to_num = up_to_num
    else:
      self.up_to_max = True

  def set_data(self, pd_data):
    if not pd_data.empty:# and isinstance(
      self.data = pd_data
    else:
      raise ValueError(
        "The last column of pd_data should be string as class label")

  def get_majority_num(self):
    total_data = self.data.values.tolist()
    labelCont = Counter(self.data[self.data.columns[-1]].values)
    majority_num = max(labelCont.values())
    return majority_num

  def run(self):
    """
    run smote
    """

    def get_ngbr(data_no_label, knn):
      rand_sample_idx = random.randint(0, len(data_no_label) - 1)
      rand_sample = data_no_label[rand_sample_idx]
      distance, ngbr = knn.kneighbors(rand_sample.reshape(1, -1))
      # while True:
      rand_ngbr_idx = random.randint(0, len(ngbr))
      #   if distance[rand_ngbr_idx] == 0:
      #     continue  # avoid the sample itself, where distance ==0
      #   else:
      return data_no_label[rand_ngbr_idx], rand_sample

    total_data = self.data.values.tolist()
    labelCont = Counter(self.data[self.data.columns[-1]].values)
    print(type(labelCont))
    majority_num = max(labelCont.values())
    for label, num in labelCont.items():
      if num < majority_num:
        to_add = majority_num - num
        last_column = self.data[self.data.columns[-1]]
        data_w_label = self.data.loc[last_column == label]
        data_no_label = data_w_label[self.data.columns[:-1]].values
        if len(data_no_label) < self.neighbor:
          num_neigh = len(data_no_label)
        else:
          num_neigh = self.neighbor
        knn = NN(n_neighbors=num_neigh).fit(data_no_label)
        for _ in range(to_add):
          rand_ngbr, sample = get_ngbr(data_no_label, knn)
          new_row = []
          for i, one in enumerate(rand_ngbr):
            gap = random.random()
            new_row.append(max(0, sample[i] + (
            sample[i] - one) * gap))
          new_row.append(label)
          total_data.append(new_row)
    return pd.DataFrame(total_data)
