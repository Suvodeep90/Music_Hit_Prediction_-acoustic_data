# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 11:57:58 2018

@author: suvod
"""

import os
import pandas as pd
from sklearn import preprocessing
import featureSelector
import learner

class dataProcessor():
    
    def __init__(self):
        self.header_ignore = "$"
        self.normalization_ignore = "*"
        self.class_s = "<"
        self.class_label = None
        self.dict_req_col = {}
        self.headers = None
        self.processed_headers = None
        self.excel_data = None
        self.dependent = None
        self.model = None
        self.clf = None
        self.df = pd.DataFrame(data = None)

    def dataProcess(self, file, data_path, model,feature_selector,normalize = False, missingValueTreatment = False, featureSelect = False):
        self.excel_data = pd.read_csv(file)
        self.model = model
        self.headers = self.excel_data.columns.values.tolist()
        for __header in self.headers:
            if self.header_ignore in __header:
                self.excel_data.drop([__header], axis = 1, inplace = True)
            if self.normalization_ignore in __header:
                self.df[__header] = self.excel_data[__header]
                self.excel_data.drop([__header], axis = 1, inplace = True)
            if self.class_s in __header:
                self.class_label = __header
        self.processed_headers = self.excel_data.columns.values.tolist()
        if missingValueTreatment:
            self.excel_data = pd.DataFrame(data = self.imputation(self.excel_data), columns = self.processed_headers)
        if normalize:
            self.dependent = self.excel_data[self.class_label]
            self.excel_data.drop([self.class_label], axis = 1, inplace = True)
            self.excel_data = pd.concat([self.DataNormalize(self.excel_data),self.df], axis=1)
            self.excel_data[self.class_label] = self.dependent
        if featureSelect:
            model = learner.learner(2)
            self.clf = model.selectedLearner(self.model)
            self.excel_data = self.featureSelction(self.excel_data,feature_selector)
        if self.excel_data.isnull().values.any():
            print("There is blank cells, please check..")
            print(self.excel_data.isnull().sum())
        self.excel_data.to_pickle(os.path.join(data_path, "processed_data.pkl"))
        self.excel_data.to_csv(os.path.join(data_path, "processed_data.csv"))
        
    def DataNormalize(self,df):
        x = df.values #returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        x_scaled = min_max_scaler.fit_transform(x)
        return pd.DataFrame(x_scaled, columns = df.columns)
    
    def imputation(self,df):
        imp = preprocessing.Imputer(missing_values='NaN', strategy='most_frequent', axis=0, verbose=0, copy=True)
        return imp.fit_transform(df)
    
    def featureSelction(self,df,feature_selector):
        if feature_selector == 'consistency_subset':
            return featureSelector.featureSelection(df,self.clf).consistency_subset()
        elif feature_selector == 'selectkBest':
            return featureSelector.featureSelection(df,self.clf).selectkBest()
        elif feature_selector == 'recursive_feature_selector':
            return featureSelector.featureSelection(df,self.clf).recursive_feature_selector()
        else:
            raise ValueError("Wrong Argument passed in feature_selector") 