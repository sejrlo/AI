from sentdexNeuralnetwork import *

fashion_mnist_labels = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot',
    }

model = Neural_Network.load("fashion_mnist.model")

X, y = load_dataset(("test",), "fashion_mnist_images")

X=X["test"]
y=y["test"]


X = (X.astype(np.float32) - 127.5)/127.5

X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])

predictions = model.predict(X)

for prediction in predictions:
    print(fashion_mnist_labels[prediction])

