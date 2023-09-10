import numpy as np
import matplotlib
import nnfs 
from nnfs.datasets import spiral_data

class Layer_Dense():
    def __init__(self, n_inputs, n_neurons, activation_function):

        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.activation_function = activation_function 

    def forward(self, inputs):
        self.output = self.activation_function.forward(np.dot(inputs, self.weights) + self.biases)
     
    def __str__(self):
        return f"Weights: {self.weights}\nBiases: {self.biases}"

class Activation_ReLU():
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        return self.output

class Activation_Softmax():
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values/np.sum(exp_values, axis=1, keepdims=True)
        return self.output

class Neural_Network():
    def __init__(self, inputs, hiddenlayers, outputs):
        self.inputs = inputs
        self.outputs = outputs
        if len(hiddenlayers) == 0: self.layers = [Layer_Dense(inputs, len(outputs), Activation_ReLU())] 
        else:
            self.layers = [Layer_Dense(inputs, hiddenlayers[0], Activation_ReLU())]

            for i in range(1, len(hiddenlayers)):
                self.layers.append(Layer_Dense(hiddenlayers[i-1], hiddenlayers[i], Activation_ReLU()))
            
            self.layers.append(Layer_Dense(hiddenlayers[-1],len(outputs), Activation_Softmax()))

    def test(self, input, targets):
        self.layers[0].forward(input)
        for i in range(1,len(self.layers)):
            self.layers[i].forward(self.layers[i-1].output)

        predictions = np.argmax(self.layers[-1].output, axis=1)
        accuracy = np.mean(predictions == targets)


        print(self.layers[-1].output)
        print("loss:", self.loss(self.layers[-1].output, targets))
        print("acc:", accuracy)

    def loss(self, outputs, targets):
        clipped_outs = np.clip(outputs, 1e-7, 1-1e-7)
        if len(targets.shape) == 1:
            neg_log = -np.log(clipped_outs[range(len(outputs)), targets])
        else:
            neg_log = -np.log(np.sum(clipped_outs*targets, axis=1))

        return np.mean(neg_log)




nnfs.init()

X, Y = spiral_data(100, 3)

nn = Neural_Network(2, [3], ["1","2","3"])

nn.test(X, Y)

