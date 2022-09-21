import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import tensorflow_hub as hub

TRAINDIR='D:\Machine learning\projects\dog Breed Classification/archive/train'
classes=os.listdir(TRAINDIR)

IMAGESHAPE=(160,160)

X_train=[]
y_train=[]
y=-1
for breed in classes:
    path=os.path.join(TRAINDIR,breed)
    y=y+1
    for img in os.listdir(path):
        image=cv2.imread(os.path.join(path,img))
        image=cv2.resize(image,IMAGESHAPE)
        X_train.append(image)
        y_train.append(y)

X_train=np.array(X_train)
X_train_scaled=X_train/255

TESTDIR='D:\Machine learning\projects\dog Breed Classification/archive/test'


X_test=[]
y_test=[]
y=-1
for breed in classes:
    path=os.path.join(TESTDIR,breed)
    y=y+1
    for img in os.listdir(path):
        image=cv2.imread(os.path.join(path,img))
        image=cv2.resize(image,IMAGESHAPE)
        X_test.append(image)
        y_test.append(y)

X_test=np.array(X_test)
X_test_scaled=X_test/255

y_train=np.array(y_train)
y_test=np.array(y_test)

feature_extractor_model="https://tfhub.dev/google/imagenet/mobilenet_v2_100_160/classification/5"
pretrained_model_without_top_layer=hub.KerasLayer(feature_extractor_model,input_shape=IMAGESHAPE+(3,),trainable=False)

# data_augmentation = keras.Sequential(
#   [
#     keras.layers.experimental.preprocessing.RandomFlip("horizontal", 
#                                                  input_shape=(160, 
#                                                               160,
#                                                               3)),
#     keras.layers.experimental.preprocessing.RandomRotation(0.1),
#     keras.layers.experimental.preprocessing.RandomZoom(0.1),
#   ]
# )

model=tf.keras.Sequential([
    # data_augmentation,
    pretrained_model_without_top_layer,
    layers.Dense(2000,activation='relu'),   
    layers.Dropout(0.2),     
    layers.Dense(1000,activation='relu'),
    layers.Dense(70)    
])

model.compile(
  optimizer="SGD",
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['acc'])

model.fit(X_train_scaled,y_train,epochs=15)

print(model.evaluate(X_test_scaled,y_test))

model.save('D:\Machine learning\projects\dog Breed Classification/archive\mymodel')