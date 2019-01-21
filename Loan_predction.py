#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 11:48:04 2018

@author: akash
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plot

df = pd.read_csv('train.csv')
df.dtypes
# Quick data exploration

# For Contineous ] Variable
description = df.describe()

# For categorical variable
catagorical_variable = df.dtypes.loc[df.dtypes == 'object'].index
print(catagorical_variable)

df[catagorical_variable].apply(lambda x: len(x.unique()))

# Univariate Analysis
df['Loan_Status'].value_counts()/df.shape[0]

# Biavariate Analysis
# 1) Categorical Vs Categorical

# 2) Both Contineous
df.plot('LoanAmount','Loan_Amount_Term',kind='scatter')



"""
----------------------------- Boolean Indexing -----------------------------
mode(df[‘Gender’].astype(str))
if you want to filter values of a column based on conditions 
from another set of columns.
Following Code will give Data of "Female" 
Loan applicants who are "Not Graduate" & got Loan
"""

df.loc[(df["Gender"]=="Female") & 
         (df["Education"]=="Not Graduate") & 
         (df["Loan_Status"]=="Y"), ["Gender","Education","Loan_Status"]]


"""
---------------------------- Apply Function --------------------------------
Apply returns some value after passing each 
row/column of a data frame with some 
function (in our case it is num_missing).
"""

def num_missing(x):
  return sum(x.isnull())

print("Missing values per column:")
print(df.apply(num_missing, axis=0))

print("\nMissing values per row:")
print(df.apply(num_missing, axis=1).head())

"""
-------------------------- Imputing Missing Values -------------------------
fillna() function is used for updating missing values with the overall 
mean, mode andmedian of the column.
"""

from scipy.stats import mode

df['Gender'].fillna(mode(df['Gender'].astype(str)).mode[0], inplace=True)
df['Married'].fillna(mode(df['Married'].astype(str)).mode[0], inplace=True)
df['Dependents'].fillna(mode(df['Dependents'].astype(str)).mode[0], inplace=True)
df['Self_Employed'].fillna(mode(df['Self_Employed'].astype(str)).mode[0], inplace=True)
df['Credit_History'].fillna(mode(df['Credit_History'].astype(str)).mode[0], inplace=True)

df = df.dropna(how='any',axis=0)


"""
------------------------ Pivot Table -------------------------------------
Pandas can be used to create MS Excel style pivot tables.
Here we will create pivot table for LoanAmount 
We can impute it using mean amount of each Gender, Married 
and Self_Employed group

values in crosstab should be Coutineous variables and indexs can be categorical
"""
impute_grps1 = df.pivot_table(values=["LoanAmount", "Loan_Amount_Term"], 
                             index=["Gender","Married","Self_Employed"], 
                             aggfunc=np.mean)
print(impute_grps1)

impute_grps2 = df.pivot_table(values=["LoanAmount", "Loan_Amount_Term"], 
                             index=["Education","Loan_Status","Credit_History"], aggfunc=np.mean)
print(impute_grps2)


impute_grps3 = df.pivot_table(values=["Total_Applicant_Income"], 
                             index=["Loan_Status","Credit_History"], aggfunc=np.mean)
print(impute_grps3)

impute_grps4 = df.pivot_table(values=["LoanAmount","Total_Applicant_Income"], 
                             index=["Loan_Status","Credit_History"], aggfunc=np.mean)
print(impute_grps4)

"""
------------------------ Crosstabs ---------------------------------------
This function is used to get an initial “feel” (view) of the data.
In this case, “Credit_History” is expected to affect the loan status significantly.
This can be tested using cross-tabulation
"""

def percConvert(ser):
  return ser/float(ser[-1])
  
pd.crosstab(df["Credit_History"],df["Loan_Status"],margins=True).apply(percConvert, axis=1)

pd.crosstab(df["Education"],df["Loan_Status"],margins=True).apply(percConvert, axis=1)

pd.crosstab(df['Gender'],df['Loan_Status'],margins=True).apply(percConvert, axis = 1)

pd.crosstab(df['Self_Employed'],df['Loan_Status'],margins=True).apply(percConvert, axis = 1)

pd.crosstab(df['Dependents'],df['Loan_Status'],margins=True).apply(percConvert, axis = 1)

pd.crosstab(df['Property_Area'],df['Loan_Status'],margins=True).apply(percConvert, axis = 1)

"""
----------------------------- Ploting --------------------------------------
We can plot Boxplot and Histograms using pandas to understand data dependency

Column for boxplot and hist should be Contineous variable 
"""

df.boxplot(column='LoanAmount', by = 'Loan_Status' )
df.hist(column='LoanAmount', by = 'Loan_Status' )

df.boxplot(column='ApplicantIncome', by = 'Credit_History' )
df.hist(column='LoanAmount', by = 'Loan_Status' )

df["Total_Applicant_Income"] = df["ApplicantIncome"]+df["CoapplicantIncome"]

df.boxplot(column='Total_Applicant_Income', by = 'Loan_Status' )
df.hist(column='LoanAmount', by = 'Loan_Status' )



X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

X.shape
y.shape

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
X[:, 4] = labelencoder_X.fit_transform(X[:, 4])
X[:, -1] = labelencoder_X.fit_transform(X[:, -1])

onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [2])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [4])
X = onehotencoder.fit_transform(X).toarray()

onehotencoder = OneHotEncoder(categorical_features = [-1])
X = onehotencoder.fit_transform(X).toarray()


labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


"""
---------------------------------------- SVM Linear ---------------------------------------------
"""
from sklearn.svm import SVC
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X,y)

y_pred = classifier.predict(X)

from sklearn.metrics import accuracy_score, confusion_matrix
accuracy_score(y, y_pred)
cf = confusion_matrix(y, y_pred)


"""
---------------------------------------- SVM Non-Linear ---------------------------------------------
"""

from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X,y)

y_pred = classifier.predict(X)

from sklearn.metrics import accuracy_score, confusion_matrix
accuracy_score(y, y_pred)
cf = confusion_matrix(y, y_pred)













