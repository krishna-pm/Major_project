import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

def importdata():
	balance_data = pd.read_csv(
'../eGeMAPSv02_adcn.csv',
	sep= ',', header = None)
	
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


def train_using_gini(X_train, X_test, y_train):

	# Creating the classifier object
	clf_gini = RandomForestClassifier(n_estimators=30,criterion='gini',random_state=1,max_depth=3) 
	# Performing training
	clf_gini.fit(X_train, y_train)
	return clf_gini

def tarin_using_entropy(X_train, X_test, y_train):

	# Decision tree with entropy
	clf_entropy = RandomForestClassifier(n_estimators=30,criterion='entropy',random_state=100,max_depth=3)

	# Performing training
	clf_entropy.fit(X_train, y_train)
	return clf_entropy

 
def prediction(X_test, clf_object):

	# Predicton on test with giniIndex
	y_pred = clf_object.predict(X_test)
	print("Predicted values:")
	print(y_pred)
	return y_pred
	
# Function to calculate accuracy
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

# Driver code
def main():
	
	# Building Phase
	data = importdata()
	X, Y, X_train, X_test, y_train, y_test = splitdataset(data)
	clf_gini = train_using_gini(X_train, X_test, y_train)
	clf_entropy = tarin_using_entropy(X_train, X_test, y_train)
	
	# Operational Phase
	print("Results Using Gini Index:")
	
	# Prediction using gini
	y_pred_gini = prediction(X_test, clf_gini)
	cal_accuracy(y_test, y_pred_gini)
	
	print("Results Using Entropy:")
	# Prediction using entropy
	y_pred_entropy = prediction(X_test, clf_entropy)
	cal_accuracy(y_test, y_pred_entropy)

	
	
	
# Calling main function
if __name__=="__main__":
	main()

