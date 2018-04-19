# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 11:51:43 2018

@author: suvod
"""

import os
import dataProcessor
import learner
import warnings

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    data_loc = 'data'
    dataSet = "unique_songDataSet.csv"
    cwd = os.getcwd()
    data_path = os.path.join(cwd, data_loc)
    file_path = os.path.join(data_path, dataSet)
    """
    ++++++++++++++++++++++++++++++++++++
    algorithm for recursive_feature_selector:
        Available options are -
        1) RF
        2) DT
    ++++++++++++++++++++++++++++++++++++
    """
    algo = 'RF'
    """
    ++++++++++++++++++++++++++++++++++++
    Feature Selector:
        Available options are -
        1) consistency_subset
        2) selectkBest
        3) recursive_feature_selector
    ++++++++++++++++++++++++++++++++++++
    """
    feature_selector = 'consistency_subset'
    dProcessor = dataProcessor.dataProcessor()
    """
    ++++++++++++++++++++++++++++++++++++
    dataProcessing parameters:
        file_path: Full Path of the dataset
        data_path: Folder path where the dataset exists
        algo: required only when using recursive_feature_selector as feature selector
        feature_selector = feature selector
        next 3 parameters are for whether to use Normalization, Missing value treatment and if feature selection will be done
    ++++++++++++++++++++++++++++++++++++
    """
    dProcessor.dataProcess(file_path, data_path, algo, feature_selector, True, True, False)
    """
    ++++++++++++++++++++++++++++++++++++
    cross validation parameter:
        nFold = Integer value depending on number of fold you want to create
    ++++++++++++++++++++++++++++++++++++
    """
    nFold = 10
    model = learner.learner(nFold)
    """
    ++++++++++++++++++++++++++++++++++++
    Learner to Run:
        Available options are -
        1) Naive Bayes: NB
        2) Decision Tree: DT
        3) Random Forest: RF
        4) Support Vector Machine: SVM
        5) Logistic Regression: LR
        6) Neural Network: MLP
        7) Our Implementation of Neural Network: NN
        8) AdaBoost: ADA
        9) Tree Based Naive Bayes with depth 2: NBL2
        10) Tree Based Naive Bayes with depth 3: NBL3
        11) Our Implementation of Naive Bayes: NBO
    ++++++++++++++++++++++++++++++++++++
    """
    _learner = "NN"
    result = model.train(_learner)
    print(result)
