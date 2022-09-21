import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from sklearn.metrics import confusion_matrix

model=keras.models.load_model("D:\Machine learning\projects\dog Breed Classification/archive\mymodel")

TESTDIR='D:\Machine learning\projects\dog Breed Classification/archive/test'

IMAGESHAPE=(160,160)
classes=os.listdir(TESTDIR)

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

y_test=np.array(y_test)

predictions=model.predict(X_test_scaled)
predicted_classes=np.argmax(predictions,axis=1)

print(predicted_classes[:5])
print(y_test[:5])

print(confusion_matrix(predicted_classes,y_test))

def predict_breed(image_path):
    image=cv2.imread(image_path)
    image=cv2.resize(image,IMAGESHAPE)
    image_array=np.array(image)
    image_scaled=image_array/255
    image_scaled=image_scaled.reshape(-1,160,160,3)
    pred = model.predict(image_scaled)
    return classes[np.argmax(pred)]


prediction_path="D:\Machine learning\projects\dog Breed Classification/archive/for predictions"


for sample in os.listdir(prediction_path):
    path=os.path.join(prediction_path,sample)
    image=cv2.imread(path)
    image=cv2.resize(image,(700,700))
    cv2.imshow(predict_breed(path),image)
    cv2.waitKey(0) 
    

