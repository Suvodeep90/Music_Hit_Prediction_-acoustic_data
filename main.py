# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 11:51:43 2018

@author: suvod
"""

import os
import pandas as pd
import dataProcessor
import learner

if __name__ == "__main__":
    data_loc = 'data'
    dataSet = "songDataSet.csv"
    cwd = os.getcwd()
    data_path = os.path.join(cwd, data_loc)
    file_path = os.path.join(data_path, dataSet)
    dProcessor = dataProcessor.dataProcessor()
    dProcessor.dataProcess(file_path,data_path,True,True,False)
    model = learner.learner()
    model.train('DT')
    