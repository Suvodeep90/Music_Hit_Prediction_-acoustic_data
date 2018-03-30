import numpy as np
import pandas as pd

# Implemented following http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/


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
        self.epsilon = 0.01         # Learning Rate
        self.reg_lambda = 0.01      # Regularization Strength

    def calculate_loss(self):
        W1, b1, W2, b2 = self.W1, self.b1, self.W2, self.b2

        # Forward propagation
        z1 = self.X.T.dot(self.W1) + self.b1
        a1 = np.tanh(z1)
        z2 = a1.dot(self.W2) + self.b2
        exp_scores = np.exp(z2)
        probs = exp_scores.T / np.sum(exp_scores, axis=1)

        # Calculate loss
        correct_logprobs = -np.log(probs)
        data_loss = np.sum(correct_logprobs)

        # Regularize loss
        data_loss += self.reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
        return 1./self.train_size * data_loss

    # Learns parameters for the neural network and returns the model.
    # hidden_layer_size: number of nodes in hidden layer
    # iterations: number of iterations through training data for gradient descent
    # print_loss: (boolean) prints loss every 1000 iterations
    def fit(self, X, y, hidden_layer_size, iterations, print_loss):
        self.X = X
        self.y = y
        self.train_size = X.shape[0]   # Assuming dataframe with rows as number of samples
        self.num_features = X.shape[1]
        self.hidden_layer_size = hidden_layer_size
        self.output_dim = y.shape[0]

        # Randomly initialize parameters to random values. These will be learned.
        np.random.seed(0)
        self.W1 = np.random.randn(self.train_size, self.hidden_layer_size) / np.sqrt(self.train_size)
        self.b1 = np.zeros((1, self.hidden_layer_size))
        self.W2 = np.random.randn(self.hidden_layer_size, self.output_dim) / np.sqrt(self.hidden_layer_size)
        self.b2 = np.zeros((1, self.output_dim))

        # Gradient descent. For each batch:
        for i in range(0, iterations):
            # Forward propagation:
            z1 = self.X.T.dot(self.W1) + self.b1
            a1 = np.tanh(z1)
            z2 = a1.dot(self.W2) + self.b2
            exp_scores = np.exp(z2)
            probs = exp_scores.T / np.sum(exp_scores, axis=1)

            # Backpropagation:
            delta3 = probs
            delta3 -= 1
            dW2 = a1.T.dot(delta3.T)
            db2 = np.sum(delta3, axis=0)
            delta2 = delta3.T.dot(self.W2.T) * (1 - np.power(a1, 2))
            dW1 = np.dot(X, delta2)
            db1 = np.sum(delta2, axis=1)

            # Adding regularization terms:
            dW2 += self.reg_lambda * self.W2
            dW1 += self.reg_lambda * self.W1

            # Parameter updates:
            self.W1 += -self.epsilon * dW1
            self.b1[0] = pd.Series(self.b1[0]) + self.epsilon * db1
            self.W2 += -self.epsilon * dW2
            self.b2[0] = pd.Series(self.b2[0]) + self.epsilon * db2

            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss()))

        return self

    # Untested - will need to change several matrix operations for this part to compile.
    # Should resemble other forward propagation code.
    def predict(self, X):
        W1, b1, W2, b2 = self.W1, self.b1, self.W2, self.b2

        # Forward propagation
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1)

        return np.argmax(probs, axis=1)

