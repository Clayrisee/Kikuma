from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # so that it runs on a mac

def preprocess_img(f_img):
    img = image.load_img(f_img, target_size=(224, 224))
    final_img = image.img_to_array(img)
    final_img_arr = np.expand_dims(final_img, axis=0)
    images = np.vstack([final_img_arr])
    return images

def get_model():
    MODEL_PATH = 'model/kikuma_model.h5'
    model = load_model(MODEL_PATH)
    return model

def predict(fname):
    img = preprocess_img(f_img=fname)
    model = get_model()
    predict_img_label = model.predict(img)
    return predict_img_label

if __name__ == '__main__':
    import pprint
    import sys

    file_name = sys.argv[1]
    results = predict(file_name)
    print(results)