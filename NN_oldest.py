# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 19:55:32 2018

@author: suvod
"""

import numpy as np
import pandas as pd
import pdb
import math
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

        # Gradient Descent Parameters
        self.epsilon = 0.001       # Learning Rate
        self.reg_lambda = 1e-6      # Regularization Strength

    def calculate_loss(self):
        W1, b1, W2, b2 = self.W1, self.b1, self.W2, self.b2

        # Forward propagation
        z1 = self.W1.T.dot(self.X) + self.b1
        a1 = self.relu_activation(z1)
        z2 = self.W2.T.dot(a1) + self.b2
        exp_scores = np.exp(z2)
        probs = exp_scores.T / np.sum(exp_scores, axis=1,keepdims = True)

        # Calculate loss
        correct_logprobs = -np.log(probs)
        data_loss = np.sum(correct_logprobs)

        # Regularize loss
        data_loss += self.reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
        print(1./self.train_size * data_loss)
        return 1./self.train_size * data_loss

    # Learns parameters for the neural network and returns the model.
    # hidden_layer_size: number of nodes in hidden layer
    # iterations: number of iterations through training data for gradient descent
    # print_loss: (boolean) prints loss every 1000 iterations
    def fit(self, X, y, hidden_layer_size, iterations, print_loss):
        N = 1965
        y = y
        self.X = np.matrix(X)
        self.y = y
        self.train_size = X.shape[0]   # Assuming dataframe with rows as number of samples
        self.num_features = X.shape[1]
        self.hidden_layer_size = hidden_layer_size
        self.output_dim = 2

        # Randomly initialize parameters to random values. These will be learned.
        np.random.seed(0)
        tf.reset_default_graph()
        w1 = tf.get_variable("w1", shape=[self.num_features, self.hidden_layer_size], initializer=tf.contrib.layers.xavier_initializer(uniform=True,seed=None,dtype=tf.float64))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.W1 = np.matrix(sess.run(w1))
        tf.reset_default_graph()
        wb1 = tf.get_variable("wb1", shape=[self.hidden_layer_size, 1], initializer=tf.contrib.layers.xavier_initializer(uniform=True,seed=None,dtype=tf.float64))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.b1 = np.matrix(sess.run(wb1))
        tf.reset_default_graph()
        w2 = tf.get_variable("w2", shape=[self.hidden_layer_size, self.output_dim], initializer=tf.contrib.layers.xavier_initializer(uniform=True,seed=None,dtype=tf.float64))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.W2 = np.matrix(sess.run(w2))
        tf.reset_default_graph()
        wb2 = tf.get_variable("wb2", shape=[self.output_dim, 1], initializer=tf.contrib.layers.xavier_initializer(uniform=True,seed=None,dtype=tf.float64))
        #print(self.train_size,self.num_features,self.hidden_layer_size,self.output_dim)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.b2 = np.matrix(sess.run(wb2))
        print(self.b1.shape)
        # Gradient descent. For each batch:
        for i in range(0, iterations):
                 
            z1 = self.W1.T.dot(self.X) + self.b1
            a1 = self.relu_activation(z1)
            z2 = self.W2.T.dot(a1) + self.b2
            exp_scores = np.exp(z2)
            probs = exp_scores / np.sum(exp_scores, axis=1)
            # Backpropagation:
            print(self.X.shape,self.W1.shape,self.b1.shape,z1.shape,a1.shape,self.W2.shape,self.b2.shape,z2.shape,exp_scores.shape,probs.shape)
            delta3 = probs
            delta3[y.astype(int),range(len(y))] -= 1
            print(delta3.shape)
            dW2 = delta3.dot(a1.T)
            db2 = np.sum(delta3, axis=1)
            print(db2)
            delta2 = delta3.dot(self.W2.T) * (1 - np.power(a1, 2))
            dW1 = X.T.dot(delta2.T)
            db1 = np.sum(delta2.T, axis=1)
            # Adding regularization terms:
            dW1 = dW1.T
            dW2 = dW2.T
            dW2 += self.reg_lambda * self.W2
            dW1 += self.reg_lambda * self.W1
            # Parameter updates:
            self.W1 += -self.epsilon * dW1
            self.b1[0] = (self.b1[0]) + self.epsilon * db1
            self.W2 += -self.epsilon * dW2
            self.b2[0] = (self.b2[0]) + self.epsilon * db2
            if print_loss and i % 100 == 0:
                print("Loss after iteration %i: %f" % (i,i))
#            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++")
#            print("weights Size:",self.W1.shape,self.W2.shape)
#            print("weight update Size:",dW1.shape,dW2.shape)
#            print("Probability Shape:", probs.shape)
#            print("delta Shape:",delta2.shape,delta3.shape)
#            print("Bias update Size:",db1.shape,db2.shape)
#            print("Bias Size:",self.b1.shape,self.b2.shape)
#            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++")
        pdb.set_trace()
        return self

    # Untested - will need to change several matrix operations for this part to compile.
    # Should resemble other forward propagation code.
    def predict(self, X):
        W1, b1, W2, b2 = self.W1, self.b1, self.W2, self.b2
        b1 = np.zeros((self.hidden_layer_size, X.shape[0]))
        b2 = np.zeros((self.output_dim, X.shape[0]))
        #pdb.traceback()
        # Forward propagation
        X = X.T
        z1 = self.W1.dot(X) + b1
        a1 = self.relu_activation(z1)
        z2 = self.W2.dot(a1) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1,keepdims = True)
        print(probs.shape)
        return np.argmax(np.array(probs.T),axis = 1)

    def relu_activation(self,data_array):
        return np.maximum(data_array, 0)
    def tanh_activation(self,data_array):
        return np.tanh(data_array)
    def sigmoid(self,z, first_derivative=False):
        if first_derivative:
            return z*(1.0-z)
        return 1.0/(1.0+np.exp(-z))

    def tanh(self, z, first_derivative=True):
        if first_derivative:
            return (1.0-z*z)
        return (1.0-np.exp(-z))/(1.0+np.exp(-z))