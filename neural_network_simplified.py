import numpy as np


class NNClassifier:

    def __init__(self):
        self.X = None
        self.y = None
        self.epoch = None
        self.learning_rate = None
        self.input_layer_neurons = None
        self.hidden_layer_neurons = None
        self.output_neurons = None

        # Weight and bias initialization
        self.wh = None
        self.bh = None
        self.wout = None
        self.bout = None

    # Sigmoid Function
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Derivative of Sigmoid
    @staticmethod
    def derivatives_sigmoid(x):
        return x * (1 - x)

    def fit(self, x, y, hidden_layer_neurons, learning_rate, epoch):
        # Variable initialization
        self.X = x
        self.y = y
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.hidden_layer_neurons = hidden_layer_neurons
        self.input_layer_neurons = self.X.shape[1]
        self.output_neurons = 1

        # weight and bias initialization
        self.wh = np.random.uniform(size=(self.input_layer_neurons, self.hidden_layer_neurons))
        self.bh = np.random.uniform(size=(1, self.hidden_layer_neurons))
        self.wout = np.random.uniform(size=(self.hidden_layer_neurons, self.output_neurons))
        self.bout = np.random.uniform(size=(1, self.output_neurons))

        for i in range(self.epoch):
            # Forward Propagation
            hidden_layer_input1 = np.dot(self.X, self.wh)
            hidden_layer_input = hidden_layer_input1 + self.bh
            hidden_layer_activations = sigmoid(hidden_layer_input)
            output_layer_input1 = np.dot(hidden_layer_activations, self.wout)
            output_layer_input = output_layer_input1 + self.bout
            output = sigmoid(output_layer_input)

            # Backpropagation
            e = y - output
            slope_output_layer = derivatives_sigmoid(output)
            slope_hidden_layer = derivatives_sigmoid(hidden_layer_activations)
            d_output = e * slope_output_layer
            error_at_hidden_layer = d_output.dot(self.wout.T)
            d_hidden_layer = error_at_hidden_layer * slope_hidden_layer
            self.wout += hidden_layer_activations.T.dot(d_output) * self.learning_rate
            self.bout += np.sum(d_output, axis=0, keepdims=True) * self.learning_rate
            self.wh += X.T.dot(d_hidden_layer) * self.lr
            self.bh += np.sum(d_hidden_layer, axis=0, keepdims=True) * self.learning_rate

    def predict(self, x):
        self.X = x
        hidden_layer_input1 = np.dot(self.X, self.wh)
        hidden_layer_input = hidden_layer_input1 + self.bh
        hidden_layer_activations = sigmoid(hidden_layer_input)
        output_layer_input1 = np.dot(hidden_layer_activations, self.wout)
        output_layer_input = output_layer_input1 + self.bout
        output = sigmoid(output_layer_input)
        return output
