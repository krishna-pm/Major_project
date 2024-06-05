from keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder



data  = pd.read_csv('../emobase_adcn.csv',sep= ',', header = None)
X = data.values[:, 0:987]
Y = data.values[:, 988]
X = np.asarray(X).astype('float32')
Y=np.asarray(Y)
labelencoder=LabelEncoder()
Y=labelencoder.fit_transform(Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)
y_train=to_categorical(y_train)


model = load_model('./audio_classification_emobase.hdf5')

predict_x=model.predict(X_test) 
classes_x=np.argmax(predict_x,axis=1)

conf_matrix=confusion_matrix(y_test,classes_x)
print(conf_matrix)

TP = conf_matrix[0][0]
FP = conf_matrix[0][1]
FN = conf_matrix[1][0]
TN = conf_matrix[1][1]

# Calculate the FMR and FNMR
FMR = FP / (FP + TN)
FNMR = FN / (FN + TP)

# Print the results
print("FMR: ", FMR)
print("FNMR: ", FNMR)
