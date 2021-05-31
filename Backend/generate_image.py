from model.model import Model
import os
import base64
import io
from PIL import Image

IMG_PATH = os.path.join(os.getcwd(),'test_images','acne-cystic-11.jpg')
encoded_img =''
with open(IMG_PATH, "rb") as image2string:
    encoded_img = base64.b64encode(image2string.read())
print(encoded_img)
print(type(encoded_img))

model = Model()

prediction= model.predict(encoded_img)
print(prediction)