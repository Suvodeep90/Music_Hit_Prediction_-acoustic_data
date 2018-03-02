# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 12:07:05 2018

@author: suvod
"""

import os
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model


class learner():
    
    def __init__(self):
        self.data_X = None
        self.data_y = None
        self.clf = None
        self.data_loc = "data"
        self.class_s = "<"
        self.class_label = None
        self.cwd = os.getcwd()
        self.data_path = os.path.join(self.cwd, self.data_loc)
        self.file_path = os.path.join(self.data_path, "processed_data.pkl")
        self.data = pd.read_pickle(self.file_path)
        self.headers = self.data.columns.values.tolist()
        for __header in self.headers:
             if self.class_s in __header:
                self.class_label = __header
        self.columns = self.data.columns.values.tolist()
        self.data_y = self.data[self.class_label]
        self.data.drop([self.class_label], axis = 1, inplace = True)
        self.data_X = self.data
        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(
                self.data_X, self.data_y, test_size=0.33, random_state=42)      
    
    def train(self):
        print("Model training Starting>>>>>>>>>>>>")
        self.clf = tree.DecisionTreeClassifier(criterion = 'entropy')
        self.clf.fit(self.train_X,self.train_y)
        predicted = self.clf.predict(self.test_X)
        print(metrics.classification_report(self.test_y, predicted, digits=3))
        res = metrics.precision_recall_fscore_support(self.test_y, predicted)
        return res
    