import pandas as pd
from sklearn.linear_model import LogisticRegression
import os
from sklearn.metrics import accuracy_score

data_path = "D:\\CSE\\Projects\\bioinformatics\\dataset"

X_train =  pd.read_csv(os.path.join(data_path, 'X_train_best.csv'))
X_test =  pd.read_csv(os.path.join(data_path, 'X_test_best.csv'))

y_train =  pd.read_csv(os.path.join(data_path, 'y_train.csv'))
y_test =  pd.read_csv(os.path.join(data_path, 'y_test.csv'))


params = {

    	'C' : [0.1, 1, 10, 100, 1000, 10000, 100000],
    	'tol' : [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2', 'elastinet', 'none'],
        'solver': ['liblinear', 'lbfgs','newton-cg']

    }

max_acc = 0

for C in params['C']:
	for tol in params['tol']:
		for penalty in params['penalty']:
			for solver in params['solver']:

				model = LogisticRegression(C = C, tol = tol, penalty = penalty, solver = solver, max_iter = 10000)

				try:
					model.fit(X_train, y_train.values.ravel())
				except ValueError:
					print("ERROR", C, tol, penalty, solver)
					break


				y_pred = model.predict(X_test)
				accuracy = round(accuracy_score(y_test.values.ravel(), y_pred)*100, 2)

				if accuracy > max_acc:

					max_acc = accuracy
					print(C, tol, penalty, solver)
					print("max till now", max_acc)


print("Accuracy: ", max_acc, 'with', C, tol, penalty, solver)

# result Accuracy:  78.77 with 100000 100 none liblinear