# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import ExcelWriter
from pandas import ExcelFile
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

df_with_na = pd.read_excel('01 Data/01 Breast Cancer Data in Excel.xlsx', sheetname='Sheet1', header=0)
rows_with_na = []

# Dropna function did not work on this dataframe. So the code below was used.
for i in range(df_with_na.shape[0]):
    if(sum(np.isnan(df_with_na.iloc[i,:]))>0):
        rows_with_na.append(i)

df = df_with_na.drop(rows_with_na)
df.Class[df.Class==2] = 0
df.Class[df.Class==4] = 1

X = df.iloc[:,1:10] # First column is ID and is excluded from the analysis.
y = df.iloc[:,10]

test_frac = 0.1
seed = 101
no_of_runs = 1000

import statsmodels.api as sm
import statsmodels.formula.api as smf

def display_matrix(data, col_names, index_names):
    """Converts Matrix into data frame and display's it in nice format."""
    df_to_display = pd.DataFrame(data, index=index_names, columns=col_names)
    print(df_to_display)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_frac, random_state=seed)
model = sm.Logit(y_train, X_train).fit()
y_pred = model.predict(X_test)
#confusion_matrix_results = confusion_matrix(y_test, y_pred)
#accuracy = metrics.accuracy_score(y_test, y_pred)
