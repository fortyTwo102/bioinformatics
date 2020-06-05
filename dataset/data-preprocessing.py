import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
import pickle
import time

# loading the dataset 

print("Running Data-Preprocessing Script!-------------------------------------------------------------------")
print("loading the dataset...")
time.sleep(1)
data = pd.read_csv('Indian Liver Patient Dataset (ILPD).csv')
print(data)

# adding feature names
data.columns = ['age', 'gender', 'total_bilirubn', 'direct_bilirubin', 'alk_phos',\
				'sgpt', 'sgot', 'total_protein', 'albumin', 'ratio_al_gl', 'target']


# standardizing data format - Gender

print("Cleaning up some columns...")
data['gender'] = data['gender'].replace({'Male': 0, 'Female': 1})

# standardizing data format - Target
data['target'] = data['target'].replace({1: 0, 2: 1})

# this shows if there are any invalid 'NaN' values in the data
# print(np.any(np.isnan(data)))


# splitting it to train and test data
X, y = data.iloc[:, :-1],  data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/4, random_state = 2)

# removing invalid values
X_train = X_train.replace(" ",np.NaN)
X_test = X_test.replace(" ",np.NaN)

imp = SimpleImputer(missing_values=np.NaN, strategy='mean')
imp.fit(X_train)
X_train = pd.DataFrame(imp.transform(X_train))
X_test = pd.DataFrame(imp.transform(X_test))


# X_train.to_csv("X-train-raw.csv", index = False)
# y_train.to_csv("y-train-raw.csv", index = False)
# X_test.to_csv("X-test-raw.csv", index = False)
# y_test.to_csv("y-test-raw.csv", index = False)

# feature scaling
print("Scaling the dataset...")
time.sleep(1)
scaler = preprocessing.StandardScaler()
scaler.fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train))
X_test = pd.DataFrame(scaler.transform(X_test))
pickle.dump(scaler, open("scaler-obj.p", "wb"))


# adding columns
X_train.columns = ['age', 'gender', 'total_bilirubn', 'direct_bilirubin', 'alk_phos','sgpt', 'sgot', 'total_protein', 'albumin', 'ratio_al_gl']
X_test.columns = ['age', 'gender', 'total_bilirubn', 'direct_bilirubin', 'alk_phos','sgpt', 'sgot', 'total_protein', 'albumin', 'ratio_al_gl']

# saving 
# data.to_csv('dataset.csv', index = False)

X_train.to_csv('X_train.csv', index = False)
X_test.to_csv('X_test.csv', index = False)

y_train.to_csv('y_train.csv', index = False)
y_test.to_csv('y_test.csv', index = False)



# print(np.any(np.isnan(data))) # to check again

print("..Done!")