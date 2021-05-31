from tensorflow.keras.models import load_model
# import cv2
import os
import numpy as np
import pickle
import pandas as pd
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input as resnet_preprocess
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input as inception_preprocess
from tensorflow.keras.applications.xception import Xception, preprocess_input as xception_preprocess
from tensorflow.keras.applications.nasnet import NASNetLarge, preprocess_input as nasnet_preprocess
from feature_extractor import extractor_model
from tensorflow.keras.preprocessing.image import load_img,img_to_array

with open(r'static\dog_breeds_category.pickle', 'rb') as handle:
    dog_breeds = pickle.load(handle)
# print(dog_breeds)    

feature_extractor_path = r'static\feature_extractor.h5'
model_path = r'static\dogbreed.h5'

feature_extractor_model = extractor_model()
predictor_model = load_model(model_path)
# print(feature_extractor.summary())
# print(predictor.summary())

def predictor(image_name): # here image is file name 
    base_path = r'static\images'
    path = os.path.join(base_path,image_name)
    img = load_img(path, target_size=(331,331))
    # print(path)
    # img = cv2.resize(img,(331,331))
    img = img_to_array(img)
    img = np.expand_dims(img,axis = 0)
    features = feature_extractor_model.predict(img)
    prediction = predictor_model.predict(features)*100
    prediction = pd.DataFrame(np.round(prediction,1),columns = dog_breeds).transpose()
    prediction.columns = ['values']
    prediction  = prediction.nlargest(5, 'values')
    return(prediction)

    






# img = cv2.imread(r'C:\Users\Abhishek\Desktop\dog_breed classifier\static\images\sample.jpg')
# img = cv2.resize(img,(331,331))
# print(img.shape)