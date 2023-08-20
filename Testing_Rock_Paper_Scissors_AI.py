import os
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
import numpy as np
import random as r

model = tf.keras.models.load_model('rps.h5')

test_images=[]
dir = input("dir to a picture or press enter to test for 10 random images of each type: ")

path = os.path.join(dir)
try: 
    image.load_img(path, target_size=(150, 150))
    test_images=[path]
    print(path)
except Exception as e:
    print(e)
 
    main_path = os.path.join('tmp/rps-test-set/')

    main_path_directories = os.listdir(main_path)

    for dircetory in main_path_directories:
        images = os.listdir(os.path.join(main_path,dircetory))
        for i in range(10):
            while True:
                index=r.randint(0,len(images)-1)
                if not images[index] in test_images:
                    test_images.append(os.path.join(main_path, dircetory, images[index]))
                    break

    #print(test_images, "\n",len(test_images))



for fn in test_images:
 
  # predicting images
  path = fn
  img = image.load_img(path, target_size=(150, 150))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)

  images = np.vstack([x])
  classes = model.predict(images, batch_size=10)
  print(fn)
  print(classes)