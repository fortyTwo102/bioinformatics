import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# loading the dataset 
data = pd.read_csv('Indian Liver Patient Dataset (ILPD).csv')

# adding feature names
data.columns = ['age', 'gender', 'total_bilirubn', 'direct_bilirubin', 'alk_phos',\
				'sgpt', 'sgot', 'total_protiens', 'albumin', 'ratio_al_gl', 'target']


# standardizing data format - Gender
data['gender'] = data['gender'].replace({'Male': 0, 'Female': 1})

# standardizing data format - Target
data['target'] = data['target'].replace({1: 0, 2: 1})

# splitting it to train and test data
train, test = train_test_split(data, test_size = 1/4, random_state = 2)

# feature scaling
scaler = preprocessing.StandardScaler()
train = pd.DataFrame(scaler.fit_transform(train))
test = pd.DataFrame(scaler.transform(test))

# saving 
data.to_csv('dataset.csv', index = False)
train.to_csv('train.csv', index = False)
test.to_csv('test.csv', index = False)
