import os
import pickle
from tkinter import *

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

l1 = ['Male', 'Female']


def predict(query):
    print(query)

    data_path = "D:\\CSE\\Projects\\bioinformatics\\dataset"
    X_train = pd.read_csv(os.path.join(data_path, 'X_train_best.csv'))
    y_train = pd.read_csv(os.path.join(data_path, 'y_train.csv'))

    # the features were learned by the Genetic Algorithm
    query = np.array(query).reshape(1, -1)

    # padding to scale
    query = np.insert(query, 2, 0)
    query = np.insert(query, 4, 0)
    query = np.insert(query, 7, 0)
    query = np.insert(query, 9, 0)

    scaler = pickle.load(open(os.path.join(data_path, 'scaler-obj.p'), 'rb'))
    query = scaler.transform([query])

    query = query[0]
    print(query)
    # delete the padding

    query = np.delete(query, 2, 0)
    query = np.delete(query, 3, 0)
    query = np.delete(query, 5, 0)
    query = np.delete(query, 6, 0)

    query = query.reshape(1, -1)

    model = LogisticRegression(C=100000, tol=0.1, penalty='l1', solver='liblinear', max_iter=100000, random_state=18318)
    model.fit(X_train, y_train.values.ravel())

    safe, unsafe = model.predict_proba(query)[0]
    return round(unsafe * 100, 2)


def prediction():
    t1.config(state='normal')
    if gender.get() == 'Male':
        g = 1
    else:
        g = 0

    l2 = [age.get(), g, dirbil.get(), sgpt.get(), sgot.get(), alb.get()]

    result = predict(l2)

    t1.delete("1.0", END)
    t1.insert(END, 'Probability of ' + Name.get() + ' Having Liver Disease is ' + str(result) + '%')
    t1.config(state='disable')


root = Tk()
root.configure(background='silver')
root.title('Disease Prediction')

Name = StringVar()

gender = StringVar()
gender.set("Select Gender")

age = IntVar()
# totbil = DoubleVar()
dirbil = DoubleVar()
sgpt = DoubleVar()
sgot = DoubleVar()
alb = DoubleVar()
# alkphos = DoubleVar()
# algl = DoubleVar()
# prot = DoubleVar()

w2 = Label(root, justify=RIGHT, text="Disease Predictor using Machine Learning", fg="teal", bg="silver")
w2.config(font=("Helvetica", 24, "bold italic"))
w2.grid(row=1, column=0, columnspan=2, padx=100)
w2.grid(row=2, column=0, columnspan=2, padx=100)

NameLb = Label(root, text="Name of the Patient:", fg="Black", bg="silver")
NameLb.config(font=("Helvetica", 16, "bold italic"))
NameLb.grid(row=5, column=0, pady=15, sticky=E)

genLb = Label(root, text="Gender", fg="teal", bg="silver")
genLb.config(font=("Helvetica", 16, "bold italic"))
genLb.grid(row=7, column=0, pady=15, sticky=E)

S1Lb = Label(root, text="Age", fg="teal", bg="silver")
S1Lb.config(font=("Helvetica", 15, "bold italic"))
S1Lb.grid(row=6, column=0, pady=10, sticky=E)

# S2Lb = Label(root, text="Total Bilirubin", fg="teal", bg="silver")
# S2Lb.config(font=("Helvetica", 15, "bold italic"))
# S2Lb.grid(row=8, column=0, pady=10, sticky=E)

S3Lb = Label(root, text="Direct Bilirubin", fg="teal", bg="silver")
S3Lb.config(font=("Helvetica", 15, "bold italic"))
S3Lb.grid(row=9, column=0, pady=10, sticky=E)

S4Lb = Label(root, text="SGPT", fg="teal", bg="silver")
S4Lb.config(font=("Helvetica", 15, "bold italic"))
S4Lb.grid(row=10, column=0, pady=10, sticky=E)

S5Lb = Label(root, text="SGOT", fg="teal", bg="silver")
S5Lb.config(font=("Helvetica", 15, "bold italic"))
S5Lb.grid(row=11, column=0, pady=10, sticky=E)

S6Lb = Label(root, text="Albumin", fg="teal", bg="silver")
S6Lb.config(font=("Helvetica", 15, "bold italic"))
S6Lb.grid(row=12, column=0, pady=10, sticky=E)

# S7Lb = Label(root, text="Alkaline phosphatase", fg="teal", bg="silver")
# S7Lb.config(font=("Helvetica", 15, "bold italic"))
# S7Lb.grid(row=13, column=0, pady=10, sticky=E)

# S8Lb = Label(root, text="Albumin-Globumin Ratio", fg="teal", bg="silver")
# S8Lb.config(font=("Helvetica", 15, "bold italic"))
# S8Lb.grid(row=14, column=0, pady=10, sticky=E)

# S9Lb = Label(root, text="Total Protein", fg="teal", bg="silver")
# S9Lb.config(font=("Helvetica", 15, "bold italic"))
# S9Lb.grid(row=15, column=0, pady=10, sticky=E)

lrLb = Label(root, text="Result:", fg="black", bg="silver")
lrLb.config(font=("Helvetica", 16, "bold italic"))
lrLb.grid(row=17, column=0, pady=10, sticky=E)

NameEn = Entry(root, textvariable=Name, width=30)
NameEn.config(font=("Helvetica", 14))
NameEn.grid(row=5, column=1)

OPTIONS = sorted(l1)
gen = OptionMenu(root, gender, *OPTIONS)
gen.config(font=("Helvetica", 10))
gen.grid(row=7, column=1)

S1 = Entry(root, textvariable=age)
S1.grid(row=6, column=1)

# S2 = Entry(root, textvariable=totbil)
# S2.grid(row=8, column=1)

S3 = Entry(root, textvariable=dirbil)
S3.grid(row=9, column=1)

S4 = Entry(root, textvariable=sgpt)
S4.grid(row=10, column=1)

S5 = Entry(root, textvariable=sgot)
S5.grid(row=11, column=1)

S6 = Entry(root, textvariable=alb)
S6.grid(row=12, column=1)

# S7 = Entry(root, textvariable=alkphos)
# S7.grid(row=13, column=1)

# S8 = Entry(root, textvariable=algl)
# S8.grid(row=14, column=1)

# S9 = Entry(root, textvariable=algl)
# S9.grid(row=15, column=1)

dst = Button(root, text="Predict",
             command=prediction, bg="teal", fg="white")
dst.config(font=("Helvetica", 15, "bold"))
dst.grid(row=8, column=3, padx=10)
dst.grid(row=9, column=3, padx=10)

t1 = Text(root, height=1, width=45, bg="white", fg="Black")
t1.config(font=("Ariel", 15, "bold"))
t1.config(state='disable')
t1.grid(row=17, column=1, padx=10)

root.mainloop()
