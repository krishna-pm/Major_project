import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from datetime import datetime 
from sklearn.metrics import accuracy_score

data  = pd.read_csv('../emobase_adcn.csv',sep= ',', header = None)
X = data.values[:, 0:987]
Y = data.values[:, 988]
X = np.asarray(X).astype('float32')
Y=np.asarray(Y)
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
Y=to_categorical(labelencoder.fit_transform(Y))

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)
num_labels=Y.shape[1]
print(num_labels)

model=Sequential()
###first layer
model.add(Dense(300,input_shape=(987,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###second layer
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###third layer
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))



###final layer
model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=15)
## Trianing my model

num_epochs = 90
num_batch_size = 32
checkpointer = ModelCheckpoint(filepath='./audio_classification_emobase.hdf5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
start = datetime.now()
model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, y_test), callbacks=[es,checkpointer], verbose=1)
duration = datetime.now() - start
print("Training completed in time: ", duration)

test_accuracy=model.evaluate(X_test,y_test,verbose=0)
print(test_accuracy[1])
#model.predict_classes(X_test)
predict_x=model.predict(X_test) 
classes_x=np.argmax(predict_x,axis=1)
print(classes_x)