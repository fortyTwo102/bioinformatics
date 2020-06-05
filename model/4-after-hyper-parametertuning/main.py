import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from itertools import combinations

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

import seaborn as sn
import matplotlib.pyplot as plt

X_train = pd.read_csv('X_train_best.csv')
X_test = pd.read_csv('X_test_best.csv')

y_train = pd.read_csv('y_train.csv').values.ravel()
y_test = pd.read_csv('y_test.csv').values.ravel()

print("Training the model....")
model = LogisticRegression(C = 100000, tol = 0.1, penalty = 'l1', solver = 'liblinear', max_iter = 100000, random_state = 18318)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = round(accuracy_score(y_pred, y_test)*100,2)
f1 = round(f1_score(y_test, y_pred)*100,2)


print(acc, f1) # 79.45 46.43


y_t = pd.DataFrame(y_test).replace({0: "Safe", 1: "Diseased"})
y_p = pd.DataFrame(y_pred).replace({0: "Safe", 1: "Diseased"})

confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])

sn.heatmap(confusion_matrix, annot=True, fmt = 'g')
plt.show()


