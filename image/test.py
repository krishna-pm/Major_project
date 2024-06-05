import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from PIL import Image

model = keras.models.load_model("/home/abhijith/Documents/alzhemer's/alzheimers1.h5")
if(model):
    print("loaded")


img1="/home/abhijith/Documents/alzhemer's/mild_101.jpg"
img2=Image.open(img1)
img3=np.array(img2)
img3=np.array(img3)
img3=img3/255
img3=img3.reshape(128,128,1)
img3 = tf.expand_dims(img3, axis=0)
predict=model.predict(img3)
p=np.argmax(predict)
if(p==0):
    print("no dementia")
elif(p==1):
    print("very mild dementia")
elif(p==2):
    print("mild dementia")
elif(p==3):
    print("moderate dementia")
else:
    print("error")

