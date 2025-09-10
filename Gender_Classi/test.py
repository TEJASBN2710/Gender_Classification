import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from keras.models import load_model

import cv2


classes =['Male', 'Female']
loaded_model = load_model("Model/New_ClassifyMF.h5")

# path = 'C:/Users/MOHAMMED AWAIS/Downloads/Gender cfy/SOCOFing/Altered/Altered-Medium/'
path= 'C:\\Users\\MOHAMMED AWAIS\\a_project\\Gender_Classi\\upload\\'
img='5__M_Left_index_finger.BMP'

img_size = 96
img_array = cv2.imread(path+img, cv2.IMREAD_GRAYSCALE)

img_resize = cv2.resize(img_array, (img_size, img_size))

t_data = np.array(img_resize).reshape(-1, img_size, img_size, 1)

t_data = t_data / 255.0

predictions=loaded_model.predict(np.expand_dims(t_data,0)[0])

print(predictions[0]*100, "\n", classes)
print("Prediction: ", classes[np.argmax(predictions)], f"{predictions[0][np.argmax(predictions)]*100}%")

