# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 08:43:52 2018

@author: suvod
"""

import numpy as np
import tensorflow as tf


class NNClassifier:

    def __init__(self):
        self.X = None
        self.y = None
        self.W1 = None
        self.W2 = None
        self.b1 = None
        self.b2 = None
        self.train_size = None
        self.num_features = None
        self.hidden_layer_size = None
        self.output_dim = None
        self.epsilon = 0.001
        self.reg_lambda = 1e-6


    def fit(self, X, y, hidden_layer_size, iterations, print_loss):
        N = len(X)
        Y = np.array(y)
        y = []
        self.X = np.array(X)
        print(Y)
        for i in range(len(Y)):
            if Y[i] == 0:
                y.append([1, 0])
            else:
                y.append([0, 1])
        self.y = np.array(y)
        self.train_size = X.shape[0]  
        self.num_features = X.shape[1]
        self.hidden_layer_size = hidden_layer_size
        self.output_dim = 2
        losses = []
        accuracies = []
        np.random.seed(0)
        tf.reset_default_graph()
        w1 = tf.get_variable("w1", shape=[self.num_features, self.hidden_layer_size], initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float64))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.W1 = (sess.run(w1))
        tf.reset_default_graph()
        wb1 = tf.get_variable("wb1", shape=[self.hidden_layer_size, 1], initializer=tf.contrib.layers.xavier_initializer(uniform=True,seed=None,dtype=tf.float64))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.b1 = (sess.run(wb1))
        tf.reset_default_graph()
        w2 = tf.get_variable("w2", shape=[self.hidden_layer_size, self.output_dim], initializer=tf.contrib.layers.xavier_initializer(uniform=True,seed=None,dtype=tf.float64))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.W2 = (sess.run(w2))
        tf.reset_default_graph()
        wb2 = tf.get_variable("wb2", shape=[self.output_dim, 1], initializer=tf.contrib.layers.xavier_initializer(uniform=True,seed=None,dtype=tf.float64))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.b2 = (sess.run(wb2))

        for i in range(0, iterations):
            
            index = np.arange(self.X.shape[0])[:N]
            # Forward propagation:
            z1 = np.matmul(self.X, self.W1) + self.b1.T
            a1 = self.sigmoid(z1)
            z2 = np.matmul(a1,self.W2) + self.b2.T
            exp_scores = self.sigmoid(z2)
            a2 = exp_scores            
            L = np.square(self.y[index]-a2).sum()/(2*N) + self.reg_lambda*(np.square(self.W1).sum()+np.square(self.W2).sum())/(2*N)
            losses.append([i, L])
            # Back Propagation
            # Output Layer
            delta3 = a2 - self.y[index]
            da2 = self.sigmoid(a2, first_derivative=True) 
            db2 = a1
            dW2 = db2.T.dot(delta3*da2) + self.reg_lambda*np.square(self.W2).sum()
            # Hidden Layer
            dl3_da2 = delta3 * da2
            delta2 = dl3_da2.dot(self.W2.T)
            da1 = self.sigmoid(a1, first_derivative=True)
            dW1 = self.X[index].T.dot(delta2*da1) + self.reg_lambda*np.square(self.W1).sum()
            self.W2 += -self.epsilon*dW2
            self.W1 += -self.epsilon*dW1
            if True:
                y_pred = self.inference(self.X, [self.W1, self.W2])
                y_actual = np.argmax(self.y, axis=1)
                accuracy = np.sum(np.equal(y_pred, y_actual))/len(y_actual)
                accuracies.append([i, accuracy])

            if i % 10000 == 0 & print_loss:
                print('Epoch %d\tLoss: %f Average L1 error: %f Accuracy: %f' % (i, L, np.mean(np.abs(delta3)), accuracy))
        return self

    def predict(self, X):
        # Forward propagation
        z1 = np.matmul(X, self.W1)
        a1 = self.sigmoid(z1)
        z2 = np.matmul(a1,self.W2)
        exp_scores = self.sigmoid(z2)
        probs = np.exp(exp_scores)/np.sum(np.exp(exp_scores), axis=1, keepdims=True)
        print(np.argmax(np.array(probs),axis = 1))
        return np.argmax(np.array(probs),axis = 1)


    def sigmoid(self, z, first_derivative=False):
        if first_derivative:
            return z*(1.0-z)
        return 1.0/(1.0+np.exp(-z))

    def tanh(self, z, first_derivative=True):
        if first_derivative:
            return 1.0-z*z
        return (1.0-np.exp(-z))/(1.0+np.exp(-z))

    def inference(self, data, weights):
        a1 = self.sigmoid(np.matmul(data, weights[0]))
        logits = np.matmul(a1, weights[1])
        probs = np.exp(logits)/np.sum(np.exp(logits), axis=1, keepdims=True)
        return np.argmax(probs, axis=1)