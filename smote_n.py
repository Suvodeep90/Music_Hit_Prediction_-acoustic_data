# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 15:45:50 2018

@author: suvod
"""

from __future__ import print_function, division
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE


class smote(object):
    def __init__(self, pd_data, k_neighbor=10,m_neighbors = 10 ,kind = 'regular'):
        self.data = None
        self.set_data(pd_data)
        self.k_neighbor = k_neighbor
        self.m_neighbors = m_neighbors
        self.kind = kind
        self.train_X = None
        self.train_y = None
        
    def set_data(self, pd_data):
        if not pd_data.empty:# and isinstance(
          self.data = pd_data
        else:
          raise ValueError(
            "The last column of pd_data should be string as class label")
    
    def set_train_test_data(self,pd):
        self.train_y = self.data[self.data.columns[-1]]
        self.train_X = self.data.loc[self.data.columns[:-1]]
        print(self.train_X)
        
      
    def fit_transform(self):
        """
        run smote
        """
        self.set_train_test_data(self.data)
        smote = SMOTE(ratio='auto', random_state=None, k_neighbors=self.k_neighbor, m_neighbors=self.m_neighbors, out_step=0.5, kind = self.kind)
        
        return pd.DataFrame(total_data)
