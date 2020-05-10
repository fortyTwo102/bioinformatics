import pandas as pd
import numpy as np
from collections import Counter


dataset = pd.read_csv('dataset.csv')
dataset = dataset.replace('Male', 0)
dataset = dataset.replace('Female', 1)
# print (dataset.head)

from sklearn.impute import SimpleImputer
dataset = dataset.replace(" ",np.NaN)
imp = SimpleImputer(missing_values=np.NaN, strategy='most_frequent')
imp.fit(dataset)
dataset = pd.DataFrame(imp.transform(dataset))



for column in dataset.columns:

	print (dataset[column].value_counts())


# from sklearn.preprocessing import OneHotEncoder 
# enc = OneHotEncoder(handle_unknown = 'ignore') 
# enc_X = pd.DataFrame(enc.fit_transform(X[['key', 'time_signature', 'mode']]).toarray())
# X = X.join(enc_X).drop(['key', 'time_signature', 'mode'], axis = 1)
# X = pd.DataFrame(X).fillna(0)