import csv
import pandas as pd
import numpy as np
import os

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

def predict(query, X_train, y_train):

	model = LogisticRegression(C = 100000, tol = 0.1, penalty = 'l1', solver = 'liblinear', max_iter = 100000, random_state = 13361) 
	model.fit(X_train, y_train.values.ravel())
	y_pred = model.predict(query)


	safe, unsafe = model.predict_proba(np.array(X_test.iloc[0, :]).reshape(1, -1))[0]
	return round(safe*100, 2)

data_path = "D:\\CSE\\Projects\\bioinformatics\\dataset"
X_train =  pd.read_csv(os.path.join(data_path, 'X_train_best.csv'))
y_train =  pd.read_csv(os.path.join(data_path, 'y_train.csv'))
X_test =  pd.read_csv(os.path.join(data_path, 'X_test_best.csv'))
query = np.array(X_test.iloc[0, :]).reshape(1, -1)

print(query)
print(predict(query, X_train, y_train))