# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 15:40:07 2018

@author: suvod
"""

from __future__ import division
import os
import pandas as pd
import math
import numpy
from sklearn.model_selection import train_test_split
from sklearn import metrics


class naive_bayes():
    
    def __init__(self):
        self.train_X = None
        self.train_y = None
        self.test_X = None
        self.test_y = None
        self.train_X_hit = None
        self.train_y_hit = None
        self.train_X_nothit = None
        self.train_y_nothit = None
        self.columns = None
        
        
    def set_data(self,data,train_X,train_y,test_X,test_y):
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y
        self.train_X_hit = self.train_X[train_y == 1]
        self.train_y_hit = self.train_y[train_y == 1]
        self.train_X_nothit = self.train_X[train_y == 0]
        self.train_y_nothit = self.train_y[train_y == 0]
        self.columns = data.columns.values.tolist()

        
    def nb_fit(self):
        # Summary Statistics
        sumstat_hit = pd.DataFrame()
        for i in range(len(self.train_X_hit.columns)):
            sumstat_hit[i] = ([numpy.mean(self.train_X_hit[self.columns[i]]), numpy.std(self.train_X_hit[self.columns[i]])])
        sumstat_nothit = pd.DataFrame()
        for i in range(len(self.train_X_nothit.columns)):
            sumstat_nothit[i] = ([numpy.mean(self.train_X_nothit[self.columns[i]]), numpy.std(self.train_X_nothit[self.columns[i]])])
        prob_hit = [len(self.train_X_hit)/len(self.train_X), len(self.train_X_nothit)/len(self.train_X)]
        res = self.NBClassifier(self.test_X,sumstat_hit,sumstat_nothit,prob_hit)
        return res
        
        
        
    def condProbability(self,x,mean,std):
        return ((1 / ((2*3.14)**0.5)*std)*math.exp((-(x-mean)**2)/(2*std**2)))
        


    def posteriorProbability(self,test_row,sumstat_hit,sumstat_nothit,prob_hit):
        posterior_probabilities = prob_hit[:]
        for i in range(len(test_row)):
    
            x = test_row[i]
    
            mean_hit = sumstat_hit[i][0]
            std_hit = sumstat_hit[i][1]
            cnd0 = self.condProbability(x,mean_hit,std_hit)
            posterior_probabilities[0] *= cnd0
            #print(posterior_probabilities[0])
    
            mean_nothit = sumstat_nothit[i][0]
            std_nothit = sumstat_nothit[i][1]
            cnd1 = self.condProbability(x, mean_nothit, std_nothit)
            posterior_probabilities[1] *= cnd1
            print([cnd0,cnd1])
        return posterior_probabilities
    
    
    def predict(self,test_row,sumstat_hit,sumstat_nothit,prob_hit):
        posterior_probabilities = self.posteriorProbability(test_row,sumstat_hit,sumstat_nothit,prob_hit)
        #print(posterior_probabilities)
        if 100000*posterior_probabilities[0] > 10*posterior_probabilities[1]:
            pred_label = 1
        else:
            pred_label = 0
        return pred_label
    
    
    def NBClassifier(self,test_X,sumstat_hit,sumstat_nothit,prob_hit):
        pred_y = []
        for i in range(len(test_X)):
            res = self.predict(test_X.iloc[i],sumstat_hit,sumstat_nothit,prob_hit)
            pred_y.append(res)
        return pred_y
