import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from itertools import combinations

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression


X_train = pd.read_csv('X_train_best.csv')
X_test = pd.read_csv('X_test_best.csv')

y_train = pd.read_csv('y_train.csv').values.ravel()
y_test = pd.read_csv('y_test.csv').values.ravel()


model = LogisticRegression(random_state = 2, max_iter = 10000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = round(accuracy_score(y_pred, y_test)*100,2)
f1 = round(f1_score(y_test, y_pred)*100,2)


print(acc, f1) # 78.08 40.74


