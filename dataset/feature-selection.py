import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from itertools import combinations

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

print("Running Feature Selection Script!-------------------------------------------------------------------")
max_acc = 0
headers = ['age', 'gender', 'total_bilirubn', 'direct_bilirubin', 'alk_phos',\
				'sgpt', 'sgot', 'total_protein', 'albumin', 'ratio_al_gl']

for i in range(1,11): # no. of columns at a time

	for columns in combinations(headers,i): # all combinations of i columns

		# print(columns)
		columns = list(columns)

		X_train = pd.read_csv('X_train.csv')
		X_test = pd.read_csv('X_test.csv')

		y_train = pd.read_csv('y_train.csv')
		y_test = pd.read_csv('y_test.csv')

		X_train = X_train[columns]
		X_test = X_test[columns]

		model = LogisticRegression(random_state = 2) 
		model.fit(X_train, y_train.values.ravel())

		# .values will give the values in an array. (shape: (n,1)
		# .ravel() will convert that array shape to (n, )


		y_pred = model.predict(X_test)

		# print(X_train.shape, X_test.shape)
		# accuracy = round(float((model.score(X_test, y_test)*100)),2)

		accuracy = round(accuracy_score(y_test.values.ravel(), y_pred)*100, 2)

		if accuracy > max_acc:

			max_acc = accuracy
			best_columns = columns
			print("Max Accuracy 'ill now", max_acc)



print("Best Accuracy: ", max_acc, ' is with ', best_columns)

# saving the selected dataset

X_train_best = X_train[best_columns].to_csv('X_train_best.csv', index = False) 
X_test_best = X_test[best_columns].to_csv('X_test_best.csv', index = False) 


print("...Done!")

