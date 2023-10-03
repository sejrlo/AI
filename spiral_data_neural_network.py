from sentdexNeuralnetwork import *
import numpy as np

nnfs.init()

X, y = spiral_data(samples=1000, classes=3)
#print(X, y)
y_true_data = []
training_data = []
batch_size = 8
epochs = 10001

for i in range(epochs):   
    training_data.append(np.array(X))
    y_true_data.append(np.array(y))

    

nn = Neural_Network(
    [
    Layer_Dense(2, 64, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4),
    Activation_ReLU(),
    Layer_Dropout(0.1),
    Layer_Dense(64, 3),
    Activation_Softmax(),
    ], 
    Loss_CategoricalCrossentropy(),
    Optimizer_Adam(learning_rate=0.05, decay=5e-5),
    Accuracy_Categorical(),
    )

print(nn.optimizer)
nn.test(np.array(X), np.array(y))

for epoch, (batch, y_true_batch) in enumerate(zip(training_data, y_true_data)):
    if not epoch % 100:
        nn.train(X, y, print_data=True, epoch=epoch)
    else:
        nn.train(X, y)

X_test, y_test = spiral_data(samples=100, classes=3)

nn.test(np.array(X_test), np.array(y_test))
