# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 16:27:39 2021

@author: Batur
"""
from tkinter import *
from PIL import ImageTk, Image, ImageOps
from tkinter import filedialog
from numpy import asarray
import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from tensorflow.keras.models import model_from_json
import numpy as np
import tensorflow as tf

#%%

json_file = open(r"C:\Users\Batur\model.json","r")
json_sea_file = open(r"C:\Users\Batur\modelsea.json","r")
json_buildings_file = open(r"C:\Users\Batur\modelbuildings.json","r")
json_forest_file = open(r"C:\Users\Batur\modelforest.json","r")
json_glacier_file = open(r"C:\Users\Batur\modelglacier.json","r")
json_street_file = open(r"C:\Users\Batur\modelstreet.json","r")
loaded_model_json = json_file.read()
loaded_model_sea_json = json_sea_file.read()
loaded_model_buildings_json = json_buildings_file.read()
loaded_model_forest_json = json_forest_file.read()
loaded_model_glacier_json = json_glacier_file.read()
loaded_model_street_json = json_street_file.read()
json_file.close()
json_sea_file.close()
json_buildings_file.close()
json_forest_file.close()
json_glacier_file.close()
json_street_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model_sea = model_from_json(loaded_model_sea_json)
loaded_model_buildings = model_from_json(loaded_model_buildings_json)
loaded_model_forest = model_from_json(loaded_model_forest_json)
loaded_model_glacier = model_from_json(loaded_model_glacier_json)
loaded_model_street = model_from_json(loaded_model_street_json)
loaded_model.load_weights(r"C:\Users\Batur\model.h5")
loaded_model_sea.load_weights(r"C:\Users\Batur\modelsea.h5")
loaded_model_buildings.load_weights(r"C:\Users\Batur\modelbuildings.h5")
loaded_model_forest.load_weights(r"C:\Users\Batur\modelforest.h5")
loaded_model_glacier.load_weights(r"C:\Users\Batur\modelglacier.h5")
loaded_model_street.load_weights(r"C:\Users\Batur\modelstreet.h5")

#%%

def loss(y_true,y_pred) :
    l = tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))
    return l

optimizer = tf.keras.optimizers.Adam(0.0001)

loaded_model.compile(
    optimizer=optimizer,
    loss=loss,
)
loaded_model_sea.compile(
    optimizer=optimizer,
    loss=loss,
)
loaded_model_buildings.compile(
    optimizer=optimizer,
    loss=loss,
)
loaded_model_forest.compile(
    optimizer=optimizer,
    loss=loss,
)
loaded_model_glacier.compile(
    optimizer=optimizer,
    loss=loss,
)
loaded_model_street.compile(
    optimizer=optimizer,
    loss=loss,
)

#%%

import cv2
import matplotlib.pyplot as plt

def plot_pair(X,y) :
    fig,axs = plt.subplots(1,2)
    axs[0].imshow(X.reshape(150,150),cmap='gray')
    axs[0].set_title('Grayscale')
    axs[1].imshow(y)
    axs[1].set_title('RGB')

def predict(path,model) :
    img = cv2.imread(path)
    img = cv2.resize(img,(150,150))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img = img / 255.0
    gray = gray / 255.0
    pred = model.predict(gray.reshape(1,150,150,1))
    pred = pred.reshape(150,150,3)
    #plot_pair(gray,pred)
    return pred

#%%
from tkinter.ttk import Combobox


#%%
root = Tk()
root.geometry("550x300")
canvas = Canvas(root, width = 550, height = 250)
comboval = "mountain"
def openfn():
    filename = filedialog.askopenfilename(title='open')
    return filename

def get_val_combo(event):
    global comboval
    comboval = combo.get()
    print(comboval)

def open_img():
    x = openfn()
    model = loaded_model
    print(comboval)
    if comboval == "mountain":
        model = loaded_model
        print("mountain")
    elif comboval == "sea":
        model = loaded_model_sea
        print("sea")
    elif comboval == "buildings":
        model = loaded_model_buildings
        print("buildings")
    elif comboval == "forest":
        model = loaded_model_forest
        print("forest")
    elif comboval == "glacier":
        model = loaded_model_glacier
        print("glacier")
    elif comboval == "street":
        model = loaded_model_street
        print("street")
    print(model)
    pair = predict(x,model)
    canvas.pack()
    originalimage = ImageTk.PhotoImage(Image.open(x))
    canvas.create_image(5,5,anchor=NW, image=originalimage)
    grayimage = ImageTk.PhotoImage(Image.open(x).convert('LA'))
    canvas.create_image(200,5,anchor=NW, image=grayimage)
    img = ImageTk.PhotoImage(image=Image.fromarray((pair*255).astype(np.uint8)))
    canvas.create_image(395,5,anchor=NW, image=img)
    textoriginal = Label(root,text="Original Image")
    textgray = Label(root, text="Gray Image")
    textpredict = Label(root, text="Predicted Image")
    textoriginal.place(x=5,y=160,anchor=NW)
    textgray.place(x=200,y=160, anchor=NW)
    textpredict.place(x=395, y=160,anchor=NW)
    root.mainloop()
    
 
canvas.delete("all")
btn = Button(root, text='chose image for colorization', command=open_img).place(x=350,y=250)

choices = ("mountain","buildings","forest","glacier","sea","street")
combo = Combobox(root, textvariable=StringVar(), values=choices, state="readonly")
combo.bind("<<ComboboxSelected>>", get_val_combo)
combo.place(x=5,y=200)

root.resizable(False,False)
#root.resizable(width=0,height=0)
root.mainloop()