
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from tensorflow.keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers import Dense, Conv2D ,Flatten,Dropout,MaxPool2D, BatchNormalization
from keras.utils import np_utils
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory  
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19
import keras
from PIL import Image
import matplotlib.pyplot as plt
import seaborn
from sklearn.metrics import confusion_matrix , classification_report
import os
import streamlit as st

def _predict(img, model):
  m = keras.models.load_model(model)
  img2 = img.resize((224, 224))

  image_array = np.asarray(img2)
  new_one = image_array.reshape((1, 224, 224, 3))

  y_pred = m.predict(new_one)
  print(y_pred)
  val = np.argmax(y_pred, axis = 1)
  return y_pred, val

uploaded_file = st.file_uploader(
    "Choose an image of your mouth", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    # img2 = Image.open('test.jpg')
    st.image(img, caption='Uploaded file of your mouth', use_column_width=True)

    # similarity = ssim(img, img2)
    # st.write("")
    # st.write(f'This is {similarity * 100}% histopathological image')

    # if similarity >= 0.85:
    st.write("")
    st.write("Classifying...")

    y_pred, val = _predict(img, '/content/drive/MyDrive/Oral Cancer/oral cancer-vggg19.h5')
    if val == 0:
      st.write(f'The patient has cancer.')
    else:
      st.write(f'The patient does not have cancer.')

