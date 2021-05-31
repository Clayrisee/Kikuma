import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import io
import base64
import os

class Model:
    def __init__(self, pretrained_model= os.path.join(os.getcwd(),'model', 'final_model'), use_segmentation=False):
        self.pretrained_model= pretrained_model
        self.use_segmentation=False
        self.model = self.__load_model()
    
    def __load_model(self):
        return tf.keras.models.load_model(self.pretrained_model)
    
    def __img_segmentation(self, img):
        pixel_values = img.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers=np.uint8(centers)
        labels = labels.flatten()
        segmented_img = centers[labels.flatten()]
        segmented_img = segmented_img.reshape(img.shape)
        masked_img =  np.copy(img)
        masked_img = masked_img.reshape((-1, 3))
        cluster = 2
        masked_img[labels==cluster] = [0, 0, 0]
        masked_img = masked_img.reshape(img.shape)

        return masked_img


    def __pre_processing(self, image):

        img = Image.open(io.BytesIO(base64.b64decode(image)))
        img = img.resize((224, 224), Image.ANTIALIAS)
        img = np.array(img) / 255

        if self.use_segmentation == True:
            img = self.__img_segmentation(img)

        return np.expand_dims(img, axis=0)

    def predict(self, image):

        model = self.model
        img = self.__pre_processing(image)
        prediction = model.predict(img)[0]

        return {'Acne Comedonal': float(prediction[0])*100, 'Acne Vulgaris':  float(prediction[1])*100}

