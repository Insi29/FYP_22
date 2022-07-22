#Import Library
import os
import numpy as np  
import pandas as pd
import tensorflow as tf
import cv2
from numpy import argmax
from tensorflow.python.keras.utils.generic_utils import populate_dict_with_module_objects
from keras.preprocessing import image as load_img
from keras.preprocessing import image as img_to_array
from tensorflow.python.keras.models import load_model

def wordRecogAndPredict(filename):
    #Loading Model
    model=load_model('website/src/utils/my_model.h5')
    image = cv2.imread(filename)
    #Labels
    characters = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H',
    'I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d',
    'e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    height, width, __ = image.shape
    #resizing the image to find spaces better
    image = cv2.resize(image, dsize=(width*10,height*3))
    #grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #binary
    _,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
    #dilation/Thinning
    kernel = np.ones((5,5), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=2)
    #adding GaussianBlur
    gsblur=cv2.GaussianBlur(img_dilation,(3,3),0)
    #find contours
    ctrs, hier = cv2.findContours(gsblur.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.minAreaRect(ctr)[0])
    
    pchl = list()
    m = list()
    ls=list()

    for i, ctr in enumerate(sorted_ctrs):
        # print all possible character in image
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)
        # Getting ROI
        roi = image[y-10:y+h+10, x-10:x+w+10]
        roi = cv2.resize(roi, dsize=(28,28), interpolation=cv2.INTER_AREA)
        #INTER_CUBIC-> for increasing
        #INTER_AREA -> for shrinking
        roi = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
        roi = np.array(roi)
        t = np.copy(roi)
        t = t / 255.0
        t = 1-t
        t = t.reshape(1,28,28,1)
        m.append(roi)
        pred = np.argmax(model.predict(t), axis=-1)
        pchl.append(pred)
        l = model.predict(t)
        o=np.argwhere(l>0.1)
        ans=""
        for c in range(len(o)):
            ans=ans+characters[o[c][1]]
            if(c!=len(o)-1):
                ans=ans+'/'
        ls.append(ans)
    pcw = list()
    interp = 'bilinear'
    for i in range(len(pchl)):
        ans=""
        for c in range(len(o)):
            ans=ans+characters[o[c][1]]
            if(c!=len(o)-1):
                ans=ans+'/'
        pcw.append(characters[pchl[i][0]])
    predstring = ''.join(pcw)
    #Returns Predicted String
    return(predstring)
