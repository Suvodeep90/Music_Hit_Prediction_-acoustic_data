# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 11:51:43 2018

@author: suvod
"""

import os
import pandas as pd
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
    algo = 'RF'
    dProcessor = dataProcessor.dataProcessor()
    dProcessor.dataProcess(file_path, data_path, algo,True, True, True)
    model = learner.learner()
    model.train("NB")
    