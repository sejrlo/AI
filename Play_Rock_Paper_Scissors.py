import os
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
import numpy as np
import random as r
from time import sleep
import cv2


rps_defeats = {"rock":"scissors", "paper":"rock","scissors":"paper"}

print("loading...")
model = tf.keras.models.load_model('rps.h5')

vid = cv2.VideoCapture(0)
print("done")

def who_won(AI_choice, human_choice):
    if AI_choice == human_choice:
        return "draw"
    elif rps_defeats[AI_choice] == human_choice:
        return "computer won"
    else: 
        return "You won"

def play_rock_paper_scissors(cheat = False):
    print("rock")
    sleep(1)
    print("paper")
    sleep(1)
    print("scissors\n")
    sleep(.5)
    ret, frame = vid.read()
    if ret:
        path="pic.jpg"

        cv2.imwrite(path, frame)
        computer_choice=r.choice(["rock","paper","scissors"])
        print(computer_choice)
        
        img = image.load_img(path, target_size=(150, 150))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        images = np.vstack([x])
        classes = model.predict(images, batch_size=10)
        human_choice=""
        if classes[0][0] == 1: human_choice = "paper"
        elif classes[0][1] == 1: human_choice = "rock"
        elif classes[0][2] == 1: human_choice = "scissors"
        if cheat:
            computer_choice = list(rps_defeats.keys())[list(rps_defeats.values()).index(human_choice)]
        else:
            computer_choice=r.choice(["rock","paper","scissors"])
        print(computer_choice)

        print("you chose",human_choice)

        print(who_won(computer_choice,human_choice))

    else: print("something went wrong with the camera")


_input = input("press enter to start")
if _input == " ":
    play_rock_paper_scissors(True)
else:
    play_rock_paper_scissors()

while True:
    _input = input("do you want to play again? Else enter 1 to exit: ")
    if "1" in _input:
        break
    elif " " == _input:
        play_rock_paper_scissors(True)
        continue
    
    play_rock_paper_scissors()

  

vid.release()
cv2.destroyAllWindows()

os.remove("pic.jpg")