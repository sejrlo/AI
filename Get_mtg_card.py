import scrython
import time
from PIL import Image
import requests
import io

query = input("Name a card: ")

auto = ""

try:
    time.sleep(0.05)
    card = scrython.cards.Named(exact=query)
except Exception:
    time.sleep(0.05)
    auto = scrython.cards.Autocomplete(q=query, query=query)

if auto:
    print("Did you mean?")
    for item in auto.data():
        print(item)
else:
    print(card.name())
    print(card.layout())
    print(card.rarity())
    
    try:
        card_faces = card.card_faces()
        colors = []
        for face in card_faces:
            for color in face["colors"]:
                if not color in colors: colors.append(color)
        print(colors)
    except Exception as e:
        print(card.colors())

    
    img_data = requests.get(card.image_uris(image_type="png")).content
    img = Image.open(io.BytesIO(img_data))
    img.show()
    