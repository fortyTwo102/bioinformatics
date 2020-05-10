import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.impute import SimpleImputer

# loading the dataset 
data = pd.read_csv('Indian Liver Patient Dataset (ILPD).csv')

# adding feature names
data.columns = ['age', 'gender', 'total_bilirubn', 'direct_bilirubin', 'alk_phos',\
				'sgpt', 'sgot', 'total_protein', 'albumin', 'ratio_al_gl', 'target']


# standardizing data format - Gender
data['gender'] = data['gender'].replace({'Male': 0, 'Female': 1})

# standardizing data format - Target
data['target'] = data['target'].replace({1: 0, 2: 1})

# this shows if there are any invalid 'NaN' values in the data
# print(np.any(np.isnan(data)))

# removing invalid values
data = data.replace(" ",np.NaN)
imp = SimpleImputer(missing_values=np.NaN, strategy='mean')
imp.fit(data)
data = pd.DataFrame(imp.transform(data))

# splitting it to train and test data
X, y = data.iloc[:, :-1],  data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/4, random_state = 2)

# feature scaling
scaler = preprocessing.StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train))
X_test = pd.DataFrame(scaler.transform(X_test))

# adding columns
X_train.columns = ['age', 'gender', 'total_bilirubn', 'direct_bilirubin', 'alk_phos','sgpt', 'sgot', 'total_protein', 'albumin', 'ratio_al_gl']
X_test.columns = ['age', 'gender', 'total_bilirubn', 'direct_bilirubin', 'alk_phos','sgpt', 'sgot', 'total_protein', 'albumin', 'ratio_al_gl']

# saving 
data.to_csv('dataset.csv', index = False)

X_train.to_csv('X_train.csv', index = False)
X_test.to_csv('X_test.csv', index = False)

y_train.to_csv('y_train.csv', index = False)
y_test.to_csv('y_test.csv', index = False)



# print(np.any(np.isnan(data))) # to check again
