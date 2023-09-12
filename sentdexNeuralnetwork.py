import numpy as np
import matplotlib
import nnfs 
from nnfs.datasets import spiral_data

class Layer_Dense():
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
 

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        self.dinputs = np.dot(dvalues, self.weights.T)
     
    def __str__(self):
        return f"Weights: {self.weights}\nBiases: {self.biases}"

class Activation_ReLU():
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax():
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values/np.sum(exp_values, axis=1, keepdims=True)
        return self.output

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1,1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
        
        

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)

        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)

        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]

        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

        neg_log = -np.log(correct_confidences)
        return neg_log
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape()) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true/dvalues
        self.dinputs = self.dinputs / samples

class Neural_Network():
    def __init__(self, inputs, hiddenlayers, outputs, loss_function):
        self.inputs = inputs
        self.outputs = outputs
        self.loss_function = loss_function
        if len(hiddenlayers) == 0: self.layers = [Layer_Dense(inputs, len(outputs)), Activation_ReLU()] 
        else:
            self.layers = [Layer_Dense(inputs, hiddenlayers[0]), Activation_ReLU()]

            for i in range(1, len(hiddenlayers)):
                self.layers.extend([Layer_Dense(hiddenlayers[i-1], hiddenlayers[i]), Activation_ReLU()])
            
            self.layers.extend([Layer_Dense(hiddenlayers[-1],len(outputs)), Activation_Softmax()])

    def test(self, input, targets):
        self.layers[0].forward(input)
        for i in range(1,len(self.layers)):
            self.layers[i].forward(self.layers[i-1].output)

        predictions = np.argmax(self.layers[-1].output, axis=1)
        accuracy = np.mean(predictions == targets)


        print(self.layers[-1].output)
        print("loss:", self.loss_function.calculate(self.layers[-1].output, targets))
        print("acc:", accuracy)

        




nnfs.init()

X, Y = spiral_data(100, 3)

nn = Neural_Network(2, [3], ["1","2","3"], Loss_CategoricalCrossentropy())

nn.test(X, Y)

