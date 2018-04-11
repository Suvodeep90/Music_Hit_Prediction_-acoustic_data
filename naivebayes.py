from __future__ import division
import os
import pandas as pd
import math
import numpy
from sklearn.model_selection import train_test_split
from sklearn import metrics

data_X = None
data_y = None
data_loc = "data"
class_s = "<"
class_label = None
cwd = os.getcwd()
data_path = os.path.join(cwd, data_loc)
file_path = os.path.join(data_path, "processed_data.pkl")
data = pd.read_pickle(file_path)
headers = data.columns.values.tolist()
for __header in headers:
    if class_s in __header:
        class_label = __header
columns = data.columns.values.tolist()
data_y = data[class_label]
data.drop([class_label], axis=1, inplace=True)
data_X = data
data_X = data_X[columns[0:15]]
train_X, test_X, train_y, test_y = train_test_split(
    data_X, data_y, test_size=0.33, random_state=42)

# Splittig training data into "Hit" and "Non-hit"
train_X_hit = train_X[train_y == 1]
train_y_hit = train_y[train_y == 1]
train_X_nothit = train_X[train_y == 0]
train_y_nothit = train_y[train_y == 0]

# Summary Statistics
sumstat_hit = pd.DataFrame()
for i in range(len(train_X_hit.columns)):
    sumstat_hit[i] = ([numpy.mean(train_X_hit[columns[i]]), numpy.std(train_X_hit[columns[i]])])
#for i in range(len(sumstat_hit.columns)):
#    if sumstat_hit[i][1] == 0:
#        sumstat_hit[i][1] = 0.0000000001

sumstat_nothit = pd.DataFrame()
for i in range(len(train_X_nothit.columns)):
    sumstat_nothit[i] = ([numpy.mean(train_X_nothit[columns[i]]), numpy.std(train_X_nothit[columns[i]])])
#for i in range(len(sumstat_nothit.columns)):
#    if sumstat_nothit[i][1] == 0:
#        sumstat_nothit[i][1] = 0.0000000001

prob_hit = [len(train_X_hit)/len(train_X), len(train_X_nothit)/len(train_X)]

# Calculating Class Probabilities

def condProbability(x,mean,std):
    #exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(std, 2))))
    #print( (1 / (math.sqrt(2 * math.pi) * std)) * exponent)
    answer = ((1 / ((2*3.14)**0.5)*std)*math.exp((-(x-mean)**2)/(2*std**2)))
    #print [x, mean, std, answer]
    return(answer)
    #print 1/(std *numpy.sqrt(2*numpy.pi)) * numpy.exp(-(x - mean)**2 / (2* std**2))
    #return 1/(std *numpy.sqrt(2*numpy.pi)) * numpy.exp(-(x - mean)**2 / (2* std**2))

def posteriorProbability(test_row,sumstat_hit,sumstat_nothit,prob_hit):
    posterior_probabilities = prob_hit[:]
    for i in range(len(test_row)):

        x = test_row[i]

        mean_hit = sumstat_hit[i][0]
        std_hit = sumstat_hit[i][1]
        cnd0 = condProbability(x,mean_hit,std_hit)
        posterior_probabilities[0] *= cnd0
        #print(posterior_probabilities[0])

        mean_nothit = sumstat_nothit[i][0]
        std_nothit = sumstat_nothit[i][1]
        cnd1 = condProbability(x, mean_nothit, std_nothit)
        posterior_probabilities[1] *= cnd1
        print([cnd0,cnd1])
    return posterior_probabilities


def predict(test_row,sumstat_hit,sumstat_nothit,prob_hit):
    posterior_probabilities = posteriorProbability(test_row,sumstat_hit,sumstat_nothit,prob_hit)
    print posterior_probabilities
    if 100000*posterior_probabilities[0] > 10*posterior_probabilities[1]:
        pred_label = 1
    else:
        pred_label = 0
    return pred_label


def NBClassifier(test_X,sumstat_hit,sumstat_nothit,prob_hit):
    pred_y = []
    for i in range(len(test_X)):
        res = predict(test_X.iloc[i],sumstat_hit,sumstat_nothit,prob_hit)
        pred_y.append(res)
    return pred_y


def check():
    print len(train_y[train_y == 1])
    print len(train_y[train_y == 0])
    #print len(train_X)
    #print len(train_X_hit)
    #print len(train_X_nothit)
    #print sumstat_hit
    #print sumstat_nothit
    #print sumstat_hit[20][0]
    #print data_X.columns
    a = test_X.iloc[1]
    #print a
    #print a[0]
    #print a.iloc[1]
    return 0

abc = NBClassifier(test_X,sumstat_hit,sumstat_nothit,prob_hit)
#check()

#print abc
print(metrics.classification_report(test_y, abc, digits=3))
print(metrics.confusion_matrix(test_y, abc))