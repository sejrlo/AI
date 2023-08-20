import scrython
import time
import urllib.request
import cv2
from PIL import Image
import requests
import io
import os

path = __file__

training_img_dir = "magic_training_img"

l = [i for i, ltr in enumerate(path) if ltr == "\\"]

parent_dir = path[:l[-1]]

path = os.path.join(parent_dir, training_img_dir)
   
if os.path.isdir(path):
    os.remove(training_img_dir)

os.mkdir(path)

i = 0
while i < 10000:
    card = scrython.Random()
    if "token" in card.layout(): continue
    if "transform" in card.layout(): continue
    if "modal_dfc" in card.layout(): continue

    #print(card.name())
    #print(card.layout())


    
    colors = ''.join(card.colors())

    if colors == "": colors =  "colorless" 

    card_name = card.name()
    card_name = card_name.replace("/","_")
    if not colors in os.listdir(path):
        os.mkdir(os.path.join(path,colors))
    try: 
        urllib.request.urlretrieve(card.image_uris(image_type="png"), os.path.join(path,colors,card_name + "(" + card.set_code() + ").png"))
    except Exception as e:
        print(card.name())
        print(e)

    i += 1
    if i%20 == 0: print(i)



