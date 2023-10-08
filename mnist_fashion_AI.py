from sentdexNeuralnetwork import *
import nnfs

path = 'fashion_mnist_images'
model_path = 'fashion_mnist_.parms'

nnfs.init()

print("loading data...")
X, y = load_dataset(("train", "test"), path)

print("done loading data...")

X_test = X["test"]
y_test = y["test"]

X = X["train"]
y = y["train"]

#Shuffle the lists but keep the connection between the labels (y) and the data (X) 

#Get the indices of the list of data
keys = np.array(range(X.shape[0]))

#Shuffle the indices
np.random.shuffle(keys)

#Resort the data (X) and the labels (y) after the indices
X = X[keys]
y = y[keys]

#Scale features to between -1 and 1
X = (X.astype(np.float32) - 127.5)/127.5
X_test = (X_test.astype(np.float32) - 127.5)/127.5

X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])

model = Neural_Network(
    [
        Layer_Dense(X.shape[1], 128),
        Activation_ReLU(),
        Layer_Dense(128, 128),
        Activation_ReLU(),
        Layer_Dense(128, 10),
        Activation_Softmax(),
    ],
    loss = Loss_CategoricalCrossentropy(),
    optimizer = Optimizer_Adam(decay=1e-3),
    accuracy = Accuracy_Categorical(),
    )

model.train(X, y, validation_data=(X_test, y_test), epochs=10, batch_size=128, print_every=100)


model.evaluate(X_test, y_test)

model.save_parameters(model_path)

model = Neural_Network(
    [
        Layer_Dense(X.shape[1], 128),
        Activation_ReLU(),
        Layer_Dense(128, 128),
        Activation_ReLU(),
        Layer_Dense(128, 10),
        Activation_Softmax(),
    ],
    loss = Loss_CategoricalCrossentropy(),
    accuracy = Accuracy_Categorical(),
    )

model.load_parameters(model_path)

model.evaluate(X_test, y_test)