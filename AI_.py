from sentdexNeuralnetwork import *

model = Neural_Network.load("fashion_mnist.model")

X, y =load_dataset(("test",), "fashion_mnist_images")

X=X["test"]
y=y["test"]


model.evaluate(X,y)