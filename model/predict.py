import csv
import pandas as pd
import numpy as np
import os

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

def predict(query, X_train, y_train):

	# the features were learned by the Genetic Algorithm
	query = np.array(query).reshape(1, -1)

	scaler = preprocessing.StandardScaler()
	X_train = pd.DataFrame(scaler.fit_transform(X_train))
	query = scaler.transform(query)

	model = LogisticRegression(C = 100000, tol = 0.1, penalty = 'l1', solver = 'liblinear', max_iter = 100000, random_state = 13361) 
	model.fit(X_train, y_train.values.ravel())
	y_pred = model.predict(query)


	safe, unsafe = model.predict_proba(np.array(X_test.iloc[0, :]).reshape(1, -1))[0]
	return round(safe*100, 2)

data_path = "..\\dataset"
X_train =  pd.read_csv(os.path.join(data_path, 'X_train_best.csv'))
y_train =  pd.read_csv(os.path.join(data_path, 'y_train.csv'))
query = [62, 0, 5.5, 64, 100, 0.74]

print(query)
print(predict(query, X_train, y_train))