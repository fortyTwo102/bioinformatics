import csv
import pandas as pd
import numpy as np
import os

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import pickle

def predict(query):

	print(query)

	data_path = "..\\dataset"
	X_train =  pd.read_csv(os.path.join(data_path, 'X_train_best.csv'))
	y_train =  pd.read_csv(os.path.join(data_path, 'y_train.csv'))


	# the features were learned by the Genetic Algorithm
	query = np.array(query).reshape(1, -1)

	# padding to scale 
	query = np.insert(query, 2, 0)
	query = np.insert(query, 4, 0)
	query = np.insert(query, 7, 0)
	query = np.insert(query, 9, 0)

	
	scaler = pickle.load(open(os.path.join(data_path, 'scaler-obj.p'), 'rb'))
	query = scaler.transform([query])

	query = query[0]

	# delete the padding
	query = np.delete(query, 2, 0)
	query = np.delete(query, 3, 0)
	query = np.delete(query, 5, 0)
	query = np.delete(query, 6, 0)

	query = query.reshape(1, -1)

	model = LogisticRegression(C = 100000, tol = 0.1, penalty = 'l1', solver = 'liblinear', max_iter = 100000, random_state = 13361) 
	model.fit(X_train, y_train.values.ravel())

	safe, unsafe = model.predict_proba(query)[0]
	return round(safe*100, 2)


query = [1, 0, 5.5, 64, 100, 0.74] 

print(predict(query))