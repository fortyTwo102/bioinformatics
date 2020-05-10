"""
Utility used by the Network class to actually train.

Based on:
    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py

"""
import os
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression



def get_ILPD():

    data_path = "D:\\CSE\\Projects\\bioinformatics\\dataset"
    X_train =  pd.read_csv(os.path.join(data_path, 'X_train_best.csv'))
    X_test =  pd.read_csv(os.path.join(data_path, 'X_test_best.csv'))

    y_train =  pd.read_csv(os.path.join(data_path, 'y_train.csv'))
    y_test =  pd.read_csv(os.path.join(data_path, 'y_test.csv'))

    return X_train, X_test, y_train.values.ravel(), y_test.values.ravel()  

def train_and_score(network, dataset):
    """Train the model, return test loss.

    Args:
        network (dict): the parameters of the network
        dataset (str): Dataset to use for training/evaluating

    """

    X_train, X_test, y_train, y_test = get_ILPD()
    

    if network['solver'] == 'liblinear':
        model = LogisticRegression(C = network['C'], tol = network['tol'], solver = network['solver'], penalty = network['penalty'], max_iter = 10000)

    elif network['solver'] in ['lbfgs', 'newton-cg', 'sag']:
        model = LogisticRegression(C = network['C'], tol = network['tol'], solver = network['solver'], penalty = 'l2', max_iter = 10000)
        
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)

    return score  # 1 is accuracy. 0 is loss.
