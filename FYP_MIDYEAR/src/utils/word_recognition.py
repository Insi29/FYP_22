import os
import numpy as np  #linear algebra
import pandas as pd
import tensorflow as tf
import cv2
#from tensorflow import keras
import cv2
from numpy import argmax
from tensorflow.python.keras.utils.generic_utils import populate_dict_with_module_objects
# from tensorflow.python.keras.preprocessing.image import load_img
from keras.preprocessing import image as load_img
# from tensorflow.python.keras.preprocessing.image import img_to_array
from keras.preprocessing import image as img_to_array
from tensorflow.python.keras.models import load_model

def word_recog_predict(filename):
    model=load_model(r'C:\Users\dsdfhj\Downloads\FYP_MIDYEAR\src\utils\my_model.h5')
    image = cv2.imread(filename)
    characters = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

    height, width, depth = image.shape

    #resizing the image to find spaces better
    image = cv2.resize(image, dsize=(width*10,height*3))

    #grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #binary
    ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
    print('threshold value',ret)

    #dilation/Thinning
    kernel = np.ones((5,5), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=2)

    #adding GaussianBlur
    gsblur=cv2.GaussianBlur(img_dilation,(3,3),0)

    #find contours
    ctrs, hier = cv2.findContours(gsblur.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    margin=0

    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.minAreaRect(ctr)[0])
    
    pchl = list()
    m = list()
    ls=list()
    for i, ctr in enumerate(sorted_ctrs):
        print('All possible character in image '+str(i)+' are: ',end=' ')
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)
        # Getting ROI
        roi = image[y-10:y+h+10, x-10:x+w+10]
        roi = cv2.resize(roi, dsize=(28,28), interpolation=cv2.INTER_AREA)
        #inter_cubic -> for increasing
        #inter_area  -> for shrinking
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
            print(characters[o[c][1]],end="")
            if(c!=len(o)-1):
                ans=ans+'/'
                print('/',end="")
        ls.append(ans)
        print()
        
    pcw = list()
    interp = 'bilinear'
    #fig, axs = plt.subplots(nrows=len(sorted_ctrs), sharex=True, figsize=(1,len(sorted_ctrs)))
    for i in range(len(pchl)):
        ans=""
        for c in range(len(o)):
            ans=ans+characters[o[c][1]]
            if(c!=len(o)-1):
                ans=ans+'/'

        pcw.append(characters[pchl[i][0]])
    #    axs[i].set_title('===>Character in image is: '+ls[i],x=3.3,y=0.24)
    #    axs[i].set_title('-------> predicted letter: '+characters[pchl[i][0]], x=2.5,y=0.24)
    #    axs[i].imshow(m[i], interpolation=interp)

    

    predstring = ''.join(pcw)
    #print('Predicted String: '+predstring)
    return(predstring)

#word_recog_predict("C:/Users/dsdfhj/Downloads/FYP_MIDYEAR/opencv_frame_14.png")