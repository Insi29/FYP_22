#Import all libraries
#########################################################################################
import os
import numpy as np  #linear algebra
import pandas as pd
import tensorflow as tf
import cv2
from tensorflow import keras
import cv2
from numpy import argmax
from tensorflow.python.keras.utils.generic_utils import populate_dict_with_module_objects
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

###########################################################################################
def infer_prec(img, img_size):
    img = tf.expand_dims(img, -1)          # from 28 x 28 to 28 x 28 x 1 
    img = tf.divide(img, 255)              # normalize 
    img = tf.image.resize(img,             # resize acc to the input
             [img_size, img_size])
    img = tf.reshape(img,                  # reshape to add batch dimension 
            [784])
    return img 


def predict(filename):
    LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
          'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
          'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']
    model_best=load_model(r"E:\insia\MY FILES\DOWNLOAD\opencv\Mathew\Mathew\src\utils\final.h5")
    #filename2='E:\insia\MY FILES\DOWNLOAD\opencv\Mathew\Mathew\opencv_frame_0.png'
    test2=[]
    img2 = cv2.imread(filename, 0)   # read image as gray scale       
    #print(img2.shape)   # (720, 1280)  
    img2 = infer_prec(img2, 28)  # call preprocess function 
    #print(img2.shape) 
    img2=tf.image.convert_image_dtype(img2, dtype=tf.float32, name=None)
    test2.append(img2)
    test2=np.array(test2)
    tested2=tf.convert_to_tensor(test2, dtype=tf.float32) #Converts image to Tensor
    y_pred2 = model_best.predict(tested2)
    #y_pred  # probabilities 
    # # get predicted label 
    pred2=tf.argmax(y_pred2, axis=-1).numpy() # array([8], dtype=int64)
    print('Predicted Label:\n',LABELS[pred2[0]])
    return LABELS[pred2[0]]
predict('E:\insia\MY FILES\DOWNLOAD\opencv\Mathew\Mathew\sample_image.png')