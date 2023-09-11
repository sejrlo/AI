import numpy as np


class Dense():
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.random.randn(1, output_size)
    
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.bias

    def backward(self, output_gradient, learning_rate):
        out = np.dot(output_gradient, self.weights.T)
        
        self.weights -= np.dot(self.inputs.T, output_gradient) * learning_rate
        self.bias -= output_gradient * learning_rate
        print(out, self.weights)
        return out

class Activation_function():
    def __init__(self, function, prime_function):
        self.function=function
        self.prime_function=prime_function
    
    def forward(self, inputs):
        self.inputs = inputs
        return self.function(inputs)
    
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1,1)


class Activation_SoftMax():
    def __init__(self):
        soft_max = lambda inputs: np.exp(inputs)/np.sum(np.exp(inputs))
        soft_max_prime = ""
        super().__init__(soft_max, soft_max_prime)

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.exp(inputs)/np.sum(np.exp(inputs))
        return self.output

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1,1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
            
        return self.dinputs

class Loss_function():
    def __init__(self, function, prime_function):
        self.function = function
        self.prime_function = prime_function
    
class MSE(Loss_function):
    def __init__(self):
        mse = lambda y, y_true: np.mean((y - y_true)**2)
        mse_prime = lambda y, y_true: 2*(y - y_true)/len(y_true)
        super().__init__(mse, mse_prime)
        
class CCEL(Loss_function):
    def __init__(self, function, prime_function):
        super().__init__(function, prime_function)    

class NeuralNetwork():
    def __init__(self, layers, loss_function):
        self.layers = layers
        self.loss_function = loss_function

    def run(self, inputs):
        out = inputs
        for layer in self.layers:
            layer.forward(out)
            out = layer.output

        return out

    def test(self, inputs, y_true):
        out = self.run(inputs)
        return out, self.loss_function.function(out, y_true)

    def train(self, inputs, y_true):
        out = self.run(inputs)
        output_gradient = self.loss_function.prime_function(out, y_true)
        for layer in reversed(self.layers):
            output_gradient = layer.backward(output_gradient, 1)

single_output = np.array([0.7, 0.1, 0.2])

single_output = single_output.reshape(-1,1)

print(single_output)

print(np.diagflat(single_output) - np.dot(single_output, single_output.T))

NN = NeuralNetwork([
    Dense(3,4),
    Dense(4,5),
    Dense(5,2),
    Activation_SoftMax(),

], MSE())

# for i in range(2):
#     NN.train(np.array([[2,3,4]]), np.array([[2,3]]))


# print(NN.test(np.array([[2,3,4]]), np.array([[2,3]])))


