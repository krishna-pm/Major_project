from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# load data
def importdata():
	balance_data = pd.read_csv('../eGeMAPSv02_adcn.csv',sep= ',', header = None)
	
	# Printing the dataswet shape
	print ("Dataset Length: ", len(balance_data))
	print ("Dataset Shape: ", balance_data.shape)
	
	# Printing the dataset obseravtions
	print ("Dataset: ",balance_data.head())
	return balance_data

def splitdataset(balance_data):

	# Separating the target variable
	X = balance_data.values[:, 0:87]
	Y = balance_data.values[:, 88]

	# Splitting the dataset into train and test
	X_train, X_test, y_train, y_test = train_test_split(
	X, Y, test_size = 0.3, random_state = 100)
	
	return X, Y, X_train, X_test, y_train, y_test

def train_using_xgb(X_train, X_test, y_train):

	# Creating the classifier object
    model =XGBClassifier(max_depth=12,n_estimators=500,learning_rate = 1)
    model.fit(X_train, y_train)
    return model

	
def prediction(X_test, clf_object):

	# Predicton on test with giniIndex
	y_pred = clf_object.predict(X_test)
	print("Predicted values:")
	print(y_pred)
	return y_pred

def cal_accuracy(y_test, y_pred):
	cm = confusion_matrix(y_test, y_pred)

	# extract true positive (TP), false positive (FP), false negative (FN)
	TP = cm[1, 1]
	FP = cm[0, 1]
	FN = cm[1, 0]

	# calculate FMR and FNMR
	FMR = FP / (FP + (cm[0, 0]))
	FNMR = FN / (FN + TP)

	print("False Match Rate (FMR):", FMR)
	print("False Non-Match Rate (FNMR):", FNMR)

	print("Confusion Matrix: ",
		confusion_matrix(y_test, y_pred))
	
	print ("Accuracy : ",
	accuracy_score(y_test,y_pred)*100)
	
	print("Report : ",
	classification_report(y_test, y_pred))

def main():
	
	# Building Phase
	data = importdata()
	X, Y, X_train, X_test, y_train, y_test = splitdataset(data)
	clf_xgb = train_using_xgb(X_train, X_test, y_train)
	
	
	# Operational Phase
	print("Results Using xgb Index:")
	
	# Prediction using gini
	y_pred_xgb = prediction(X_test, clf_xgb)
	cal_accuracy(y_test, y_pred_xgb)
	
	
	
	
# Calling main function
if __name__=="__main__":
	main()

