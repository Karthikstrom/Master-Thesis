# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 11:34:13 2022

@author: Karthikeyan
"""

#%% Importing packages

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.preprocessing import LabelEncoder
from sklearn import tree


#%% Importing data

df=pd.read_csv(r"C:\Users\Karthikeyan\Desktop\Github\Master-Thesis\Codebasics\salaries.csv")

#%% Removing the target 

inputs = df.drop('salary_more_then_100k',axis='columns')
target= df['salary_more_then_100k']

#%% Machine learning needs number so encodeing the strings

le_company = LabelEncoder()
le_job= LabelEncoder()
le_degree=LabelEncoder()

inputs['company_n']=le_company.fit_transform(inputs['company'])
inputs['job_n']=le_job.fit_transform(inputs['job'])
inputs['degree_n']=le_degree.fit_transform(inputs['degree'])

inputs.drop(['company','job','degree'],axis='columns',inplace=True)

#%% Build and train our classifier

model=tree.DecisionTreeClassifier()
model.fit(inputs,target)

#%%
model.score(inputs,target)
model.predict([[2,0,1]])

