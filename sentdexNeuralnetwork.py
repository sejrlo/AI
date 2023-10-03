import numpy as np
import matplotlib
import nnfs 
from nnfs.datasets import spiral_data
from random import randint
# https://towardsdatascience.com/how-to-teach-an-ai-to-play-games-deep-reinforcement-learning-28f9b920440a

class Layer_Dropout:
    def __init__(self, rate):
        self.rate = 1 - rate

    def forward(self, inputs):
        self.inputs = inputs
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate

        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, weight_regularizer_l1=0, weight_regularizer_l2=0,
                bias_regularizer_l1=0, bias_regularizer_l2=0):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

 

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        
        # L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1

        # L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights

        # L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1

        # L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases
     
        self.dinputs = np.dot(dvalues, self.weights.T)
    
    def __str__(self):
        return f"Weights: {self.weights}\nBiases: {self.biases}"

class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
    
    def predictions(self, outputs):
        return outputs

class Activation_Linear:
    def forward(self, inputs):
        self.inputs = inputs 
        self.output = inputs
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
    
    def predictions(self, outputs):
        return outputs

class Activation_Softmax:
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
    
    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)

class Activation_Sigmoid:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
    
    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output
    
    def predictions(self, outputs):
        return (outputs > .5) * 1

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)

        return data_loss, self.regularization_loss()

    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def regularization_loss(self):
        for layer in self.trainable_layers:
            regularization_loss = 0

            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))

            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)

            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))

            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)
            
            return regularization_loss

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

class Loss_BinaryCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        sample_losses = -(y_true * np.log(y_pred_clipped) + (1-y_true) * np.log(1-y_pred_clipped))

        sample_losses = np.mean(sample_losses, axis=-1)

        return sample_losses

    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        outputs = len(dvalues[0])

        clipped_dvalues = np.clip(dvalues, 1e-7, 1-1e-7)

        self.dinputs = -(y_true / clipped_dvalues - (1- y_true) / (1 - clipped_dvalues)) / outputs

        self.dinputs = self.dinputs / samples

        return self.dinputs

class Loss_MeanSquaredError(Loss):
    def forward(self, y_pred, y_true):
        sample_loss = np.mean((y_pred-y_true)**2, axis=-1)

        return sample_loss
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])

        self.dinputs = -2 * (y_true - dvalues) / outputs

        self.dinputs = self.dinputs/samples

        return self.dinputs

class Loss_AbsoluteError(Loss):
    def forward(self, y_pred, y_true):
        sample_losses = np.mean(np.abs(y_true-y_pred),axis=-1)

        return sample_losses
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])

        self.dinputs = np.sign(y_true-dvalues) / outputs
        self.dinputs = self.dinputs/ samples

        return self.dinputs

class Activation_Softmax_Loss_CategoricalCrossentropy:
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, inputs):
        self.activation.forward(inputs)
        self.output = self.activation.output

    def remember_trainable_layers(self, trainable_layers):
        self.loss.remember_trainable_layers(trainable_layers)

    def calculate(self, outputs, y_true):
        return self.loss.calculate(outputs, y_true)
    
    def backward(self, dvalues, y_true=None):
        if y_true is None:
            return
        
        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples
        return self.dinputs

class Accuracy:
    def calculate(self, predictions, y):
        comparisons = self.compare(predictions, y)
        
        accuracy = np.mean(comparisons)
        
        return accuracy

class Accuracy_Regression(Accuracy):
    def __init__(self):
        self.precission = None

    def init(self, y, reinit=False):
        if self.precission is None or reinit:
            self.precission = np.std(y) / 250
    
    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precission

class Accuracy_Categorical(Accuracy):
    def __init__(self, *, binary=False):
        self.binary = binary
    
    def init(self, y):
        pass

    def compare(self, predictions, y):
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)

        return y == predictions

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

class Neural_Network:
    def __init__(self, layers, loss_function, optimizer, accuracy):
        self.optimizer = optimizer
        self.accuracy = accuracy

        #check if last activation function and loss function can be combined
        for combination in combined_activation_loss_functions:
            if isinstance(loss_function, combination["loss"]) and isinstance(layers[-1], combination["activation"]):
                self.loss_function = combination["combined"]()
                layers.pop(-1)
                layers.append(self.loss_function)
                break
        else:
            self.loss_function = loss_function
        
        self.layers = layers
        self.trainable_layers = []
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                self.trainable_layers.append(layer)
        
        self.loss_function.remember_trainable_layers(self.trainable_layers)

    def run(self, input, training=False):
        self.layers[0].forward(input)
        for i in range(1,len(self.layers)):
            if isinstance(self.layers[i], Layer_Dropout) and not training:
                self.layers[i].output = self.layers[i-1].output
                continue
            self.layers[i].forward(self.layers[i-1].output)
        
        return self.layers[-1].output

    def test(self, input, targets, return_outs=False):
        outputs = self.run(input)

        predictions = self.layers[-1].predictions(outputs)
        accuracy = self.accuracy.calculate(predictions, targets)

        data_loss, regularization_loss = self.calculate_loss(outputs, targets)
        loss = data_loss + regularization_loss

        print(outputs[:5])
        print("loss:", loss)
        print("acc:", accuracy)

        if return_outs:
            return outputs
    
    def train(self, input, targets, print_data=False, epoch=None):
        outputs = self.run(input, training=True)
        if print_data:
            predictions = self.layers[-1].predictions(outputs)
            accuracy = self.accuracy.calculate(predictions, targets)
            data_loss, regularization_loss = self.calculate_loss(outputs, targets)
            loss = data_loss + regularization_loss
            if not epoch is None:
                print(f"epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f} (data_loss: {data_loss:.3f}, regularization_loss: {regularization_loss:.3f}), lr: {self.optimizer.current_learning_rate:.10f}")
            else:
                print(f"acc: {accuracy:.3f}, loss: {loss:.3f} (data_loss: {data_loss:.3f}, regularization_loss: {regularization_loss:.3f}), lr: {self.optimizer.current_learning_rate:.10f}")

        output_gradient = self.loss_function.backward(outputs, targets)

        for layer in reversed(self.layers):
            layer.backward(output_gradient)
            output_gradient = layer.dinputs
        
        self.optimizer.optimize(self.trainable_layers)

    def calculate_loss(self, outputs, targets):
        if hasattr(self, "combined_loss_and_activation_function"):
            return self.combined_loss_and_activation_function.calculate(outputs, targets)
        else:
            return self.loss_function.calculate(outputs, targets)
    