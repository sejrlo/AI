import numpy as np

class Layer():
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        pass

    def backward(self, output_dradient, learning_rate):
        pass

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.random.randn(1, output_size)
    
    def forward(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.bias
    
    #change to work with batches
    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(self.input.T, output_gradient)
        output = np.dot(output_gradient, self.weights.T)
        self.bias -= learning_rate * np.sum(output_gradient, axis=0)
        self.weights -= learning_rate * weights_gradient
        return output
    

class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime
    
    def forward(self, input):
        self.input = input
        return self.activation(input)
    
    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))
    
class Tanh(Activation):
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        tanh_prime  = lambda x: 1-np.tanh(x)**2
        super().__init__(tanh, tanh_prime)

class Loss_function():
    def __init__(self, loss_function, loss_function_prime):
        self.loss_function = loss_function
        self.loss_function_prime = loss_function_prime
    
class MSE(Loss_function):
    def __init__(self):
        MSE = lambda y_true, y_pred: np.mean(np.power(y_pred - y_true, 2), axis=1)
        MSE_prime = lambda y_true, y_pred: 2 * (y_pred - y_true) / np.shape(y_true)[1]
        super().__init__(MSE, MSE_prime)

class Network():
    def __init__(self, layers, loss_function, learning_rate=1.0):
        self.layers = layers
        self.loss_function = loss_function
        self.learning_rate = learning_rate

    def run(self, inputs, y_true=None):
        output = inputs
        for layer in self.layers:
            output = layer.forward(output)
        
        if y_true is None: return output

        return output, self.loss_function.loss_function(y_true, output)

    def test(self, inputs, y_true):
        if len(np.shape(inputs)) == 1:
            inputs = np.array([inputs])
        
        if len(np.shape(y_true)) == 1:
            y_true = np.array([y_true])

        return self.run(inputs, y_true)

    def training(self, inputs, y_true):
        if len(np.shape(inputs)) == 1:
            inputs = np.array([inputs])
        
        if len(np.shape(y_true)) == 1:
            y_true = np.array([y_true])

        output, error = self.run(inputs, y_true)
        grad = self.loss_function.loss_function_prime(y_true, output)
        for layer in reversed(self.layers):
            grad = layer.backward(grad, self.learning_rate)

    def batch_training(self, inputs, y_trues, batch_size):
        batchs = []
        for i in range(0, len(inputs), batch_size):
            batchs.append(inputs[i:i+batch_size])
        
        y_true_batchs = []
        for i in range(0, len(inputs), batch_size):
            y_true_batchs.append(y_trues[i:i+batch_size])


        for batch, y_true_batch in zip(batchs, y_true_batchs):
            self.training(np.array(batch), np.array(y_true_batch))
        


X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([[0], [1], [1], [0]])


network = Network([Dense(2,3), Tanh(), Dense(3,1), Tanh()], MSE(), 0.1)

training_inputs = []
training_outputs = []

batch_size = 1

for e in range(batch_size*10000):
    rn = np.random.randint(0, len(X))
    training_inputs.append(X[rn])
    training_outputs.append(Y[rn])

print(training_inputs[:10])
print(training_outputs[:10])

network.batch_training(training_inputs, training_outputs, batch_size)

for x, y in zip(X, Y):
    output, loss = network.test(x, y)
    print("input:", x)
    print("output:", output, "loss:", loss)
