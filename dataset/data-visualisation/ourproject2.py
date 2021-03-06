import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
import os

# loading the dataset 
# data = pd.read_csv('X_train.csv')

# adding feature names
# data.columns = ['age', 'gender', 'total_bilirubn', 'direct_bilirubin', 'alk_phos',\
# 				'sgpt', 'sgot', 'total_protein', 'albumin', 'ratio_al_gl', 'target']

# scaler = preprocessing.StandardScaler()
# scaler.fit_transform(data)

# features = ['age','total_bilirubn', 'direct_bilirubin', 'alk_phos','sgpt', 'sgot', 'total_protein', 'albumin', 'ratio_al_gl']
# # data['target'] = data['target'].replace({1: 'Safe', 2: 'Diseased'})





# # features in pairgrid graph

# x=sns.PairGrid(data,vars=['total_bilirubn', 'direct_bilirubin'])

# x=x.map(plt.scatter)
# plt.show()

# # standardizing data format - Gender
# data['gender'] = data['gender'].replace({'Male': 0, 'Female': 1})

# #males and females in graph
# sns.violinplot(x='gender',y='age',data=data,inner='stick',hue='gender',split=True,scale='count').set(title='males and females after standardizing',
# 															  ylabel='count')
# plt.show()

# standardizing data format - Target




# this shows if there are any invalid 'NaN' values in the data
# print(np.any(np.isnan(data)))


# # splitting it to train and test data

X_train = pd.read_csv('X_train.csv')
# X, y = data.iloc[:, :-1],  data.iloc[:, -1]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/4, random_state = 2)

# removing invalid values
X_train = X_train.replace(" ",np.NaN)
# X_test = X_test.replace(" ",np.NaN)

imp = SimpleImputer(missing_values=np.NaN, strategy='mean')
imp.fit(X_train)
X_train = pd.DataFrame(imp.transform(X_train))
# X_test = pd.DataFrame(imp.transform(X_test))


# feature scaling
scaler = preprocessing.StandardScaler()
scaler.fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train))
# X_test = pd.DataFrame(scaler.transform(X_test))
# pickle.dump(scaler, open("scaler-obj.p", "wb"))

# #X_test data after feature scaling in barplot
# sns.barplot(data=X_test,ci=None).set(title='X_test data after feature scaling')
# plt.show()

# #X_train data after feature scaling in barplot
# sns.barplot(data=X_train,ci=None).set(title='X_train data after feature scaling')
# plt.show()

# # adding columns
X_train.columns = ['age', 'gender', 'total_bilirubn', 'direct_bilirubin', 'alk_phos','sgpt', 'sgot', 'total_protein', 'albumin', 'ratio_al_gl']
# X_test.columns = ['age', 'gender', 'total_bilirubn', 'direct_bilirubin', 'alk_phos','sgpt', 'sgot', 'total_protein', 'albumin', 'ratio_al_gl']

# # total_bilirubn,direct_bilirubin,alk_phos graphs in X_train
# x=sns.PairGrid(X_train,vars=['total_bilirubn','direct_bilirubin','alk_phos'])
# x=x.map(plt.scatter)
# plt.show()



sns.set(style="white")

# Generate a large random dataset
d = X_train

# Compute the correlation matrix
corr = d.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=np.bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.show()
# # total_bilirubn,direct_bilirubin,alk_phos graphs in X_test
# x=sns.PairGrid(X_test,vars=['total_bilirubn','direct_bilirubin','alk_phos'])
# x=x.map(plt.scatter)
# plt.show()

# features =['age','total_bilirubn', 'direct_bilirubin', 'alk_phos','sgpt', 'sgot', 'total_protein', 'albumin', 'ratio_al_gl']

# for combo in combinations(features, 2):

# 	sns.jointplot(x=X_train[combo[0]], y=X_train[combo[1]],kind='reg');
# 	plt.show()

# #relation between direct bilirubn and total_bilirubn
# sns.jointplot(x=X_train['direct_bilirubin'], y=X_train['total_bilirubn'],kind='reg');
# plt.show()
# #relation between total protein and albumin
# sns.jointplot(x=X_train['total_protein'], y=X_train['albumin'],kind='reg')
# plt.show()
# #relation between ratio al gl and albumin
# sns.jointplot(x=X_train['ratio_al_gl'], y=X_train['albumin'],kind='reg');
# plt.show()


# # saving 
# data.to_csv('dataset.csv', index = False)

# X_train.to_csv('X_train.csv', index = False)
# X_test.to_csv('X_test.csv', index = False)

# y_train.to_csv('y_train.csv', index = False)
# y_test.to_csv('y_test.csv', index = False)



# #print(np.any(np.isnan(data))) # to check again
