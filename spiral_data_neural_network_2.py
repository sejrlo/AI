from sentdexNeuralnetwork import *
import numpy as np

nnfs.init()

X, y = spiral_data(samples=100, classes=2)
y = y.reshape(-1,1)

X_test, y_test = spiral_data(samples=100, classes=2)

y_test = y_test.reshape(-1,1)

nn = Neural_Network(
    [
    Layer_Dense(2, 64, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4),
    Activation_ReLU(),
    Layer_Dense(64, 1),
    Activation_Sigmoid(),
    ], 
    Loss_BinaryCrossentropy(),
    Optimizer_Adam(decay=5e-7),
    Accuracy_Categorical(binary=True),
    )

print(nn.optimizer)
nn.test(np.array(X), np.array(y))

nn.train(X, y, validation_data=(X_test, y_test), epochs=10000, print_every=100)

