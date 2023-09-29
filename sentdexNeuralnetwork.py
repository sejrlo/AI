import numpy as np
import matplotlib
import nnfs 
from nnfs.datasets import spiral_data
from random import randint
# https://towardsdatascience.com/how-to-teach-an-ai-to-play-games-deep-reinforcement-learning-28f9b920440a

class Layer_Dense():
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
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
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax():
    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

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

        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]

        
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        neg_log = -np.log(correct_confidences)
        return neg_log
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        #print("dv", dvalues[:5])
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples
        return self.dinputs

class Activation_Softmax_Loss_CategoricalCrossentropy():
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, inputs):
        self.activation.forward(inputs)
        self.output = self.activation.output

    def calculate(self, outputs, y_true):
        return self.loss.calculate(outputs, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples
        return self.dinputs

class Optimizer_Adam:
    # Initialize optimizer - set settings,
    def __init__(self, learning_rate=0.001, decay=0., epsilon = 1e-7, beta_1 = .9, beta_2 = .999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def optimize(self, layers):
        self.pre_update_params()
        for layer in layers:
            self.update_params(layer)

        self.post_update_params()

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / (1. + self.decay * self.iterations)

    # Update parameters
    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
            
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
        
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1-self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1-self.beta_1) * layer.dbiases

        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))


        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights ** 2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases ** 2

        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1

    def __str__(self):
        if self.current_learning_rate == self.learning_rate:
            return f"Lr: {self.learning_rate}\nDecay: {self.decay}\nEpsilon: {self.epsilon}\nBeta_1: {self.beta_1}\nBeta_2: {self.beta_2}"
        else:
            return f"Lr: {self.learning_rate}\nCur_Lr: {self.current_learning_rate}\nDecay: {self.decay}\nIterations: {self.iterations}\nEpsilon: {self.epsilon}\nBeta_1: {self.beta_1}\nBeta_2: {self.beta_2}"
      
class Optimizer_RMSProp:
    # Initialize optimizer - set settings,
    def __init__(self, learning_rate=0.001, decay=0., epsilon = 1e-7, rho = .9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.rho = rho
        self.epsilon = epsilon

    def optimize(self, layers):
        self.pre_update_params()
        for layer in layers:
            self.update_params(layer)

        self.post_update_params()

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / (1. + self.decay * self.iterations)

    # Update parameters
    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights ** 2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases ** 2

        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1

    def __str__(self):
        if self.current_learning_rate == self.learning_rate:
            return f"Lr: {self.learning_rate}\nDecay: {self.decay}\nEpsilon: {self.epsilon}\nRho: {self.rho}"
        else:
            return f"Lr: {self.learning_rate}\nCur_Lr: {self.current_learning_rate}\nDecay: {self.decay}\nIterations: {self.iterations}\nEpsilon: {self.epsilon}\nRho: {self.rho}"
        
class Optimizer_AdaGrad:
    # Initialize optimizer - set settings,
    def __init__(self, learning_rate=1., decay=0., epsilon = 1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    def optimize(self, layers):
        self.pre_update_params()
        for layer in layers:
            self.update_params(layer)

        self.post_update_params()

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / (1. + self.decay * self.iterations)

    # Update parameters
    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache = layer.dweights ** 2
        layer.bias_cache = layer.dbiases ** 2

        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1

    def __str__(self):
        if self.current_learning_rate == self.learning_rate:
            return f"Lr: {self.learning_rate}\nDecay: {self.decay}\nEpsilon: {self.epsilon}"
        else:
            return f"Lr: {self.learning_rate}\nCur_Lr: {self.current_learning_rate}\nDecay: {self.decay}\nIterations: {self.iterations}\nEpsilon: {self.epsilon}"

class Optimizer_SGD:
    # Initialize optimizer - set settings,
    # learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def optimize(self, layers):
        self.pre_update_params()
        for layer in layers:
            self.update_params(layer)

        self.post_update_params()

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate/(1. + self.decay * self.iterations)

    # Update parameters
    def update_params(self, layer):
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
    
            weights_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.weight_momentums = weights_updates
            layer.bias_momentums = bias_updates
        
        else:
            weights_updates -= self.current_learning_rate * layer.dweights
            bias_updates -= self.current_learning_rate * layer.dbiases

        layer.weights += weights_updates
        layer.biases += bias_updates

    def post_update_params(self):
        self.iterations += 1

    def __str__(self):
        if self.current_learning_rate == self.learning_rate:
            return f"Lr: {self.learning_rate}\nDecay: {self.decay}\nMomentum: {self.momentum}"
        else:
            return f"Lr: {self.learning_rate}\nCur_Lr: {self.current_learning_rate}\nDecay: {self.decay}\nIterations: {self.iterations}\nMomentum: {self.momentum}"

combined_activation_loss_functions = [
    {"loss":Loss_CategoricalCrossentropy, "activation": Activation_Softmax, "combined":Activation_Softmax_Loss_CategoricalCrossentropy}
    ]

class Neural_Network():
    def __init__(self, inputs, hiddenlayers, outputs, activation_function, loss_function, optimizer, last_layer_activation_function=None):
        self.inputs = inputs
        self.outputs = outputs

        if last_layer_activation_function is None: last_layer_activation_function = activation_function
        #check if last activation function and loss function can be combined
        for combination  in combined_activation_loss_functions:
            if combination["loss"] == loss_function and combination["activation"] == last_layer_activation_function:
                self.combined_loss_and_activation_function = combination["combined"]()
                break
        else:
            self.loss_function = loss_function()
        
        self.optimizer = optimizer

        if len(hiddenlayers) == 0: self.layers = [Layer_Dense(inputs, len(outputs)), activation_function()] 
        else:
            self.layers = [Layer_Dense(inputs, hiddenlayers[0]), Activation_ReLU()]

            for i in range(1, len(hiddenlayers)):
                self.layers.extend([Layer_Dense(hiddenlayers[i-1], hiddenlayers[i]), Activation_ReLU()])

            if hasattr(self, "combined_loss_and_activation_function"):
                self.layers.extend([Layer_Dense(hiddenlayers[-1], len(outputs)), self.combined_loss_and_activation_function])
            else:
                self.layers.extend([Layer_Dense(hiddenlayers[-1], len(outputs)), last_layer_activation_function()])

    def run(self, input):
        self.layers[0].forward(input)
        for i in range(1,len(self.layers)):
            self.layers[i].forward(self.layers[i-1].output)
        
        return self.layers[-1].output

    def test(self, input, targets):
        outputs = self.run(input)

        predictions = np.argmax(outputs, axis=1)
        accuracy = np.mean(predictions == targets)


        print(outputs[:5])
        print("loss:", self.calculate_loss(outputs, targets))
        print("acc:", accuracy)
    
    def train(self, input, targets, print_data=False, epoch=None):
        outputs = self.run(input)
        if print_data:
            predictions = np.argmax(outputs, axis=1)
            accuracy = np.mean(predictions == targets)
            loss = self.calculate_loss(outputs, targets)
            if not epoch is None:
                print(f"epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f}, lr: {self.optimizer.current_learning_rate}")
            else:
                print(f"acc: {accuracy:.3f}, loss: {loss:.3f}, lr: {self.optimizer.current_learning_rate}")
        if hasattr(self, "loss_function"): output_gradient = self.loss_function.backward(outputs, targets)
        else: output_gradient = self.combined_loss_and_activation_function.backward(outputs, targets)
        for layer in reversed(self.layers):
            if isinstance(layer, Activation_Softmax_Loss_CategoricalCrossentropy):
                continue
            output_gradient = layer.backward(output_gradient)
            output_gradient = layer.dinputs
        
        layers_to_optimize = []
        for layer in self.layers:
            if hasattr(layer, "weights"): layers_to_optimize.append(layer)
        self.optimizer.optimize(layers_to_optimize)

    def calculate_loss(self, outputs, targets):
        if hasattr(self, "combined_loss_and_activation_function"):
            return self.combined_loss_and_activation_function.calculate(outputs, targets)
        else:
            return self.loss_function.calculate(outputs, targets)
    
nnfs.init()

X, y = spiral_data(samples=100, classes=3)
#print(X, y)
y_true_data = []
training_data = []
batch_size = 8
""" X = [[0,0], [1,0], [0,1], [1,1]]
y = [0, 1, 1, 0] """
epochs = 10001
""" for i in range(epochs):
    batch = []
    y_true_batch = []
    for u in range(batch_size):
        d = np.random.randint(0,2,(2,))
        y_true = int(d[0] != d[1])
        batch.append(d)
        y_true_batch.append(y_true)

    training_data.append(np.array(batch))
    y_true_data.append(np.array(y_true_batch)) """

for i in range(epochs):
    """ batch = []
    y_true_batch = []
    for i in range(batch_size):
        index = randint(0, len(X)-1)
        batch.append(X[index])
        y_true_batch.append(y[index]) """
    
    training_data.append(np.array(X))
    y_true_data.append(np.array(y))

    

nn = Neural_Network(2, [64], ["0", "1", "2"], Activation_ReLU, Loss_CategoricalCrossentropy, 
                    Optimizer_Adam(learning_rate=0.05, decay=5e-7), last_layer_activation_function=Activation_Softmax)
print(nn.optimizer)
nn.test(np.array(X), np.array(y))

for epoch, (batch, y_true_batch) in enumerate(zip(training_data, y_true_data)):
    if not epoch % 100:
        nn.train(X, y, print_data=True, epoch=epoch)
    else:
        nn.train(X, y)

X_test, y_test = spiral_data(samples=100, classes=3)

nn.test(np.array(X_test), np.array(y_test))
