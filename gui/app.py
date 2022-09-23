from cProfile import label
from genericpath import isfile
import tkinter as tk
from tkinter import Canvas, Label, PhotoImage, filedialog,Text
import os
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix
from PIL import ImageTk, Image
import tkinter.font as font

model=tf.keras.models.load_model("D:\Machine learning\projects\dog Breed Classification/archive\mymodel")

root=tk.Tk()
root.wm_title("Dog Breed Identifier")
root.iconbitmap("D:\Machine learning\projects\dog Breed Classification\gui\data\Dog_Breed_Identifier_logo-no-bg.ico")
background_image=Image.open("D:\Machine learning\projects\dog Breed Classification\gui\data\OIP.jpg")
background_photo=ImageTk.PhotoImage(background_image.resize((420,350),Image.ANTIALIAS))
# bimg=Label(root,image=background_photo)
# bimg.pack()



TESTDIR='D:\Machine learning\projects\dog Breed Classification/archive/test'

IMAGESHAPE=(160,160)
classes=os.listdir(TESTDIR)

image_path=""

if os.path.isfile('save.txt'):
    with open('save.txt','r')as f:
        tempimages=f.read()
        tempimages=tempimages.split(',')
        images=[x for x in tempimages if x.strip()]

def predict_breed(image_path):
    print(image_path)
    image=cv2.imread(image_path)
    image=cv2.resize(image,IMAGESHAPE)
    image_array=np.array(image)
    image_scaled=image_array/255
    image_scaled=image_scaled.reshape(-1,160,160,3)
    pred = model.predict(image_scaled)
    return classes[np.argmax(pred)]

def addFile():

    for widget in image_frame.winfo_children():
        widget.destroy()
    
    for widget in result_frame.winfo_children():
        widget.destroy()

    global image_path
    

    filename=filedialog.askopenfilename(initialdir="/",title="Select File",
    filetypes=(("Images","*.jpg"),("all files","*.*")))

    image_path=filename

    img=Image.open(image_path)
    photo=ImageTk.PhotoImage(img.resize((420,390),Image.ANTIALIAS))
        
    label=tk.Label(image_frame,image=photo)
    label.image=photo
    label.place(x=0,y=0)

    

result_font=font.Font(family='Times New Roman',size=40)
def identify():
    for widget in result_frame.winfo_children():
        widget.destroy()

    result=predict_breed(image_path)
    result_label=tk.Label(result_frame,text=result,font=result_font)
    result_label.pack()

Canvas=tk.Canvas(root,height=600,width=600,bg="#f0eff0")
Canvas.pack()

image_frame=tk.Frame(root,bg="white")
image_frame.place(relwidth=0.6666,relheight=0.6,relx=0.15,rely=0.1)

# upload_image=Image.open("D:\Machine learning\projects\dog Breed Classification\gui\data\upload.png")
# upload_photo=ImageTk.PhotoImage(upload_image,Image.ANTIALIAS)
bimg=Label(image_frame,image=background_photo)
bimg.pack()

result_frame=tk.Frame(root,bg="white")
result_frame.place(relwidth=.6666,relheight=.1,relx=0.15,rely=0.65)

f=font.Font(family='Times New Roman',size=13)


addimage_frame=tk.Frame(root,bg="#f0eff0")
addimage_frame.place(x=145,y=550,width=100,h=50)
addimage_label=tk.Label(addimage_frame,text="Add Image",font=f)
addimage_label.place(x=0,y=0)
addimage_pic=Image.open("D:\Machine learning\projects\dog Breed Classification\gui\data\Button-Add-1-512.webp")
addimage_pic=ImageTk.PhotoImage(addimage_pic.resize((60,60),Image.ANTIALIAS))
openFile=tk.Button(root,text="Add Image",padx=5,pady=5,command=addFile,width=60,height=60,font=f,borderwidth=0,image=addimage_pic)
openFile.place(x=150,y=470)
# openFile.pack()


identify_frame=tk.Frame(root,bg="#f0eff0")
identify_frame.place(x=360,y=550,width=100,h=50)
identify_label=tk.Label(identify_frame,text="Identify?",font=f)
identify_label.place(x=0,y=0)
identify_pic=Image.open("D:\Machine learning\projects\dog Breed Classification\gui\data\search-button-png-image-free-download.png")
identify_pic=ImageTk.PhotoImage(identify_pic.resize((55,55),Image.ANTIALIAS))
Identify=tk.Button(root,text="Identify",padx=5,pady=5,command=identify,width=55,height=55,font=f,borderwidth=0,image=identify_pic)
Identify.place(x=360,y=480)
# Identify.pack()

title_font=font.Font(family="Times New Roman",size=15)
title_frame=tk.Frame(root,bg="#f0eff0")
title_frame.place(x=60,y=10,width=500,height=20)
title_label=tk.Label(title_frame,text="Know the dog breed out of 70 breeds in seconds",font=title_font)
title_label.place(x=30,y=0)

root.mainloop()

# with open('save.txt','w')as f:
#     for img in images:
#         f.write(img+',')
