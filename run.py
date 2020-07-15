import src.NeuralNetwork as NN
import numpy as np

if __name__ == "__main__":
    X = np.array([[0,0,1],
                  [0,1,1],
                  [1,0,1],
                  [1,1,1]])
    y = np.array([[0], [1], [1], [0]])
    nn = NN.NeuralNetwork(X,y)

    for i in range(1000):
        nn.feedforward()
        nn.backprop()

    print(nn.output)
