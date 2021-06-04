# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 17:41:00 2021

@author: Batur
"""

import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

#%%

X = []
y = []
train_path = r'C:\Users\Batur\Desktop\seg_train\seg_train\forest'

#%%
for i in sorted(os.listdir(train_path)) :
    sub_path = os.path.join(train_path,i)
    img = cv2.imread(sub_path)
    img = cv2.resize(img,(150,150))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    X.append(cv2.cvtColor(img,cv2.COLOR_RGB2GRAY))
    y.append(img)
X = np.asarray(X)
y = np.asarray(y)

X = X.reshape(2271,150,150,1)

#%%
def plot_pair(X,y) :
    fig,axs = plt.subplots(1,2)
    axs[0].imshow(X.reshape(150,150),cmap='gray')
    axs[0].set_title('Grayscale')
    axs[1].imshow(y)
    axs[1].set_title('RGB')
    
plot_pair(X[0],y[0])
#%%

plot_pair(X[23],y[23])
#%%
X = X / 255.0
y = y / 255.0

def model1() :
    inp = tf.keras.layers.Input(shape=(150,150,1))
    conv1 = tf.keras.layers.Conv2D(128,3,padding='same',activation='relu') (inp)
    conv2 = tf.keras.layers.Conv2D(128,3,padding='same',activation='relu') (conv1)
    convt1 = tf.keras.layers.Conv2DTranspose(128,3,padding='same',activation='relu') (conv2)
    convt2 = tf.keras.layers.Conv2DTranspose(128,3,padding='same',activation='relu') (convt1)
    convt3 = tf.keras.layers.Conv2DTranspose(128,3,padding='same',activation='relu') (convt2)
    out = tf.keras.layers.Conv2DTranspose(3,3,padding='same',activation='sigmoid') (convt3)
    
    model = tf.keras.models.Model(inp,out)
    model.summary()
    return model

model = model1()

#%%
def loss(y_true,y_pred) :
    l = tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))
    return l

optimizer = tf.keras.optimizers.Adam(0.0001)
model.compile(optimizer=optimizer,loss=loss)

hist = model.fit(X,y,epochs=15,batch_size=16,validation_split=0.2)

#%%
# SAVING THE MODEL AND WEIGHTS
import h5py
from keras.models import model_from_json

#%%
model_json = model.to_json()

with open("modelforest.json","w") as json_file:
    json_file.write(model_json)
model.save_weights("modelforest.h5")

json_file.close()

#%%
def predict(path) :
    img = cv2.imread(path)
    img = cv2.resize(img,(150,150))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img = img / 255.0
    gray = gray / 255.0
    pred = model.predict(gray.reshape(1,150,150,1))
    pred = pred.reshape(150,150,3)
    plot_pair(gray,pred)

#%%
pred = model.predict(X[0].reshape(1,150,150,1))

pred = pred.reshape(150,150,3)
plot_pair(X[0],pred)

#%%
predict('../input/intel-image-classification/seg_test/seg_test/mountain/20356.jpg')

predict('../input/intel-image-classification/seg_test/seg_test/mountain/20174.jpg')

predict('../input/intel-image-classification/seg_test/seg_test/mountain/20435.jpg')