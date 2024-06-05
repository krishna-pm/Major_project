import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import joblib

# Load data from CSV file
data  = pd.read_csv('../compare_adcn.csv',sep= ',', header = None)
X = data.values[:, 0:6372]
Y = data.values[:, 6373]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)

# Initialize individual models
rf = RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=100,max_depth=5)
xg = xgb.XGBClassifier(max_depth=12,n_estimators=300,learning_rate = 1) 
dt = DecisionTreeClassifier(criterion = "gini",random_state = 50,max_depth=5, min_samples_leaf=5) 
# Initialize the voting regressor
voting_cli = VotingClassifier(estimators=[('rf', rf), ('xgb', xg), ('dt', dt)],voting='hard')

# Fit the voting regressor to the training data
voting_cli.fit(X_train, y_train)
y_pred = voting_cli.predict(X_test)
print("Predicted values:")
print(y_pred)

# Use the voting regressor to make predictions on new data
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
print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))
	
print ("Accuracy : ", accuracy_score(y_test,y_pred)*100)
	
print("Report : ", classification_report(y_test, y_pred))

filename = 'finalized_model.sav'
joblib.dump(voting_cli, filename)

