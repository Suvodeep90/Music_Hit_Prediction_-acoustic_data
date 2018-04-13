# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 08:43:52 2018

@author: suvod
"""

import numpy as np
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
        Y = y
        y = []
        self.X = np.array(X)
        print(Y)
        for i in range(len(Y)):
            if Y[i] == 0:
                y.append([1,0])
            else:
                y.append([0,1])
        self.y = np.array(y)
        self.train_size = X.shape[0]   # Assuming dataframe with rows as number of samples
        self.num_features = X.shape[1]
        self.hidden_layer_size = hidden_layer_size
        self.output_dim = 2
        losses = []
        accuracies=[]

        # Randomly initialize parameters to random values. These will be learned.
        np.random.seed(0)
        tf.reset_default_graph()
        w1 = tf.get_variable("w1", shape=[self.num_features, self.hidden_layer_size], initializer=tf.contrib.layers.xavier_initializer(uniform=True,seed=None,dtype=tf.float64))
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
        #print(self.train_size,self.num_features,self.hidden_layer_size,self.output_dim)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.b2 = (sess.run(wb2))
        # Gradient descent. For each batch:
        for i in range(0, iterations):
            
            index = np.arange(self.X.shape[0])[:N]
            # Forward propagation:
            #pdb.set_trace()
            z1 = np.matmul(self.X, self.W1) #+ self.b1.T
            a1 = self.sigmoid(z1)
            z2 = np.matmul(a1,self.W2) #+ self.b2.T
            exp_scores = self.sigmoid(z2)
#            probs = np.exp(exp_scores)/np.sum(np.exp(exp_scores), axis=1, keepdims=True)
            a2 = exp_scores
            
            L = np.square(self.y[index]-a2).sum()/(2*N) + self.reg_lambda*(np.square(self.W1).sum()+np.square(self.W2).sum())/(2*N)

            losses.append([i,L])
            
            
            #Back Propagation
            delta3 = -(self.y[index] - a2)
            dh2_dz2 = self.sigmoid(a2, first_derivative=True) 
            db2 = a1
            dW2 = db2.T.dot(delta3*dh2_dz2) + self.reg_lambda*np.square(self.W2).sum()
            #xdb2 = np.sum(dh2_dz2.T, axis=1)
            
            dL_dz2 = delta3 * dh2_dz2
            dz2_dh1 = self.W2
            delta2 = dL_dz2.dot(dz2_dh1.T)
            dh1_dz1 = self.sigmoid(a1, first_derivative=True)
            dz1_dw1 = self.X[index]
            dW1 = dz1_dw1.T.dot(delta2*dh1_dz1) + self.reg_lambda*np.square(self.W1).sum()
            #xdb1 = np.sum(dh1_dz1.T, axis=1)
            
            self.W2 += -self.epsilon*dW2
            self.W1 += -self.epsilon*dW1
            if True: #(i+1)%1000==0:
                y_pred = self.inference(self.X, [self.W1, self.W2])
                y_actual = np.argmax(self.y, axis=1)
                accuracy = np.sum(np.equal(y_pred,y_actual))/len(y_actual)
                accuracies.append([i, accuracy])

            if (i+1)% 10000 == 0:
                print('Epoch %d\tLoss: %f Average L1 error: %f Accuracy: %f' %(i, L, np.mean(np.abs(delta3)), accuracy))
        return self

    # Untested - will need to change several matrix operations for this part to compile.
    # Should resemble other forward propagation code.
    def predict(self, X):
        W1, b1, W2, b2 = self.W1, self.b1, self.W2, self.b2
        b1 = np.zeros((self.hidden_layer_size, X.shape[0]))
        b2 = np.zeros((self.output_dim, X.shape[0]))
        #pdb.traceback()
        # Forward propagation
        z1 = np.matmul(X, self.W1)# + self.b1.T
        a1 = self.sigmoid(z1)
        z2 = np.matmul(a1,self.W2)# + self.b2.T
        exp_scores = self.sigmoid(z2)
        probs = np.exp(exp_scores)/np.sum(np.exp(exp_scores), axis=1, keepdims=True)
        print(np.argmax(np.array(probs),axis = 1))
        return np.argmax(np.array(probs),axis = 1)

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
    def inference(self,data, weights):
        a1 = self.sigmoid(np.matmul(data, weights[0]))
        logits = np.matmul(a1, weights[1])
        probs = np.exp(logits)/np.sum(np.exp(logits), axis=1, keepdims=True)
        return np.argmax(probs, axis=1)