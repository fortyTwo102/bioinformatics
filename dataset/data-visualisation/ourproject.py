import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# loading the dataset 
data = pd.read_csv('Indian Liver Patient Dataset (ILPD).csv')

# adding feature names
data.columns = ['age', 'gender', 'total_bilirubn', 'direct_bilirubin', 'alk_phos',\
				'sgpt', 'sgot', 'total_protein', 'albumin', 'ratio_al_gl', 'target']



# features in pairgrid graph

x=sns.PairGrid(data,vars=['total_bilirubn', 'direct_bilirubin', 'alk_phos',\
				'sgpt'])

x=x.map(plt.scatter)
plt.show()

# standardizing data format - Gender
data['gender'] = data['gender'].replace({'Male': 0, 'Female': 1})

#males and females in graph
sns.violinplot(x='gender',y='age',data=data,inner='stick',hue='gender',split=True,scale='count').set(title='males and females after standardizing',
															  ylabel='count')
plt.show()

# standardizing data format - Target
data['target'] = data['target'].replace({1: 0, 2: 1})



# this shows if there are any invalid 'NaN' values in the data
# print(np.any(np.isnan(data)))


# splitting it to train and test data
X, y = data.iloc[:, :-1],  data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/4, random_state = 2)

# removing invalid values
X_train = X_train.replace(" ",np.NaN)
X_test = X_test.replace(" ",np.NaN)

imp = SimpleImputer(missing_values=np.NaN, strategy='mean')
imp.fit(X_train)
X_train = pd.DataFrame(imp.transform(X_train))
X_test = pd.DataFrame(imp.transform(X_test))


# feature scaling
scaler = preprocessing.StandardScaler()
scaler.fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train))
X_test = pd.DataFrame(scaler.transform(X_test))
pickle.dump(scaler, open("scaler-obj.p", "wb"))

#X_test data after feature scaling in barplot
sns.barplot(data=X_test).set(title='X_test data after feature scaling')
plt.show()

#X_train data after feature scaling in barplot
sns.barplot(data=X_train).set(title='X_train data after feature scaling')
plt.show()

# adding columns
X_train.columns = ['age', 'gender', 'total_bilirubn', 'direct_bilirubin', 'alk_phos','sgpt', 'sgot', 'total_protein', 'albumin', 'ratio_al_gl']
X_test.columns = ['age', 'gender', 'total_bilirubn', 'direct_bilirubin', 'alk_phos','sgpt', 'sgot', 'total_protein', 'albumin', 'ratio_al_gl']

# total_bilirubn,direct_bilirubin,alk_phos graphs in X_train
x=sns.PairGrid(X_train,vars=['total_bilirubn','direct_bilirubin','alk_phos'])
x=x.map(plt.scatter)
plt.show()


# total_bilirubn,direct_bilirubin,alk_phos graphs in X_test
x=sns.PairGrid(X_test,vars=['total_bilirubn','direct_bilirubin','alk_phos'])
x=x.map(plt.scatter)
plt.show()




#print(np.any(np.isnan(data))) # to check again
