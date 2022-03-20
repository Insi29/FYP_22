import numpy as np
import tensorflow as tf
#from tensorflow import keras
from tensorflow.python.keras.utils.generic_utils import populate_dict_with_module_objects
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import KFold
from PIL import Image
from numpy import argmax
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

import matplotlib
import matplotlib.pyplot as plt

# save the final model to file
# load train and test dataset

# load train and test dataset
#plot original image
def plot_image(img_path):
	img=Image.open(img_path)
	plt.imshow(img, cmap='gray')
# load and prepare the image
def load_image(img_path):
	# load the image
	img = load_img(img_path, color_mode='grayscale', target_size=(28, 28))
  
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 1 channel
	img = img.reshape(1, 28, 28, 1)
	# prepare pixel data
	img = img.astype('float32')
	img = img / 255.0
	return img
 
# load an image and predict the class
def mnist_predict(image_path):
	# load the image
	img = load_image(image_path)
	# load model
	model_mnist=load_model(r'E:\insia\MY FILES\DOWNLOAD\opencv\Mathew\Mathew\src\utils\mnist_model.h5')
  
	# predict the class
	predict_value = model_mnist.predict(img)
	digit = argmax(predict_value)
	print('Predicted Result:',digit)
# entry point, run the example
mnist_predict('E:\insia\MY FILES\DOWNLOAD\opencv\Mathew\Mathew\sample_image.png')