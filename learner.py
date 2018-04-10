# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 12:07:05 2018

@author: suvod
"""

import os
import pandas as pd
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import neural_network
import smote
from sklearn.ensemble import AdaBoostClassifier



class learner():
    
    def __init__(self):
        self.data_X = None
        self.data_y = None
        self.doSmt = False
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
    
    def train(self, model):
        print("Model training Starting>>>>>>>>>>>>")
        self.selectedLearner(model)
        print(self.test_X.shape)
        print(self.train_X.shape)
        if self.doSmt:
            self.doSmote()
        self.clf.fit(self.train_X,self.train_y)
        predicted = self.clf.predict(self.test_X)
        print(metrics.classification_report(self.test_y, predicted, digits=3))
        print(metrics.confusion_matrix(self.test_y, predicted))
        res = metrics.precision_recall_fscore_support(self.test_y, predicted)
        return res
    
    def selectedLearner(self,model):
        if model == 'DT':
            print("Decision Tree Training")
            self.clf = tree.DecisionTreeClassifier(criterion = 'entropy')
        elif model == 'RF':
            print("Random Forest Tree Training")
            self.clf = RandomForestClassifier(criterion = 'entropy')
        elif model == 'SVM':
            print("SVM Training")
            self.clf = svm.SVC(kernel = 'linear')
        elif model == 'LR':
            print("Logistic Regression Training")
            self.clf = LogisticRegression()
        elif model == 'NB':
            print("Naive Bayes Training")
            self.clf = GaussianNB()
        elif model == 'MLP':
            print("Neural Network Training")
            self.clf = MLPClassifier(activation = 'relu', max_iter=1000)
        elif model == 'NN':
            print("Our Neural Network Training")
            self.clf = neural_network.NNClassifier()
        elif model == 'ADA':
            print("Our Neural Network Training")
            self.clf = AdaBoostClassifier()
            

    def doSmote(self):
        df = pd.concat([self.train_X,self.train_y], axis = 1)
        columnNames = self.data.columns.values.tolist()
        columnNames.append(self.class_label)
        smt = smote.smote(df,5)        
        self.data = smt.run()
        self.data.columns = columnNames
        self.train_y = self.data[self.class_label]
        self.data.drop([self.class_label], axis = 1, inplace = True)
        self.train_X = self.data