import numpy as np
from src.functions import sigmoid, sigmoid_derivative

class NeuralNetwork:
    def __init__(self, x, y):
        """
        Initialize params
        """
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1],4)
        self.weights2 = np.random.rand(4,1)
        self.y = y
        self.output = np.zeros(y.shape)

    def feedforward(self):
        """Calculate the output:
        y_hat = sig(W_2 * sig(W_1 * x + b_1) + b_2)
        """
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        """Apply the chain rule to find the derivative of the 
        loss functon with respect to the weights
        """

        derivative_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        derivative_weights1 = np.dot(self.input.T, (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        self.weights1 += derivative_weights1
        self.weights2 += derivative_weights2

