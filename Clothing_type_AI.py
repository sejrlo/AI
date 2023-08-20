import tensorflow as tf
from tensorflow import keras
import numpy as np

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print(test_labels[0:10])

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax),
])

model.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy')

model.fit(train_images, train_labels, epochs=5)

print(model.evaluate(test_images, test_labels))

predictions = model.predict(test_images)


for prediction, label in zip(predictions[0:10], test_labels[0:10]):
    max_value = max(prediction)
    index = list(prediction).index(max_value)
    
    print(index, label)

    if index != label:
        print(prediction)

