import nnfs
from nnfs.datasets import sine_data
from sentdexNeuralnetwork import *
import matplotlib.pyplot as plt


X, y = sine_data()

nnfs.init()

y_true_data = []
training_data = []
batch_size = 8
epochs = 10001

for i in range(epochs):   
    training_data.append(np.array(X))
    y_true_data.append(np.array(y))

nn = Neural_Network(
    [
    Layer_Dense(1, 64),
    Activation_ReLU(),
    Layer_Dense(64, 64),
    Activation_ReLU(),
    Layer_Dense(64, 1),
    Activation_Linear(),
    ], 
    Loss_MeanSquaredError(),
    Optimizer_Adam(learning_rate=0.005, decay=1e-3),
    Accuracy_Regression(),
    )

print(nn.optimizer)
nn.accuracy.init(y)
nn.test(np.array(X), np.array(y))

nn.train(X, y, epochs=10000, print_every=100)


X_test, y_test = sine_data()

outputs = nn.test(np.array(X_test), np.array(y_test), return_outs=True)

plt.plot(X_test, y_test)
plt.plot(X_test, outputs)

plt.show()