# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 14:08:54 2023

@author: Karthikeyan
"""

import pandas as pd

df_l=pd.DataFrame()
df_l=pd.read_csv(r'C:\Users\Karthikeyan\Desktop\Github\Master-Thesis\Main_Folder\4_Single_Step_DNN\Prediction_csv\load.csv')

df_pv=pd.DataFrame()
df_pv=pd.read_csv(r'C:\Users\Karthikeyan\Desktop\Github\Master-Thesis\Main_Folder\4_Single_Step_DNN\Prediction_csv\pv.csv')

df_pr=pd.DataFrame()
df_pr=pd.read_csv(r'C:\Users\Karthikeyan\Desktop\Github\Master-Thesis\Main_Folder\4_Single_Step_DNN\Prediction_csv\price.csv')

df_b=pd.DataFrame()
df_b=pd.read_csv(r'C:\Users\Karthikeyan\Desktop\Github\Master-Thesis\Main_Folder\4_Single_Step_DNN\Prediction_csv\baseline.csv')

df=pd.DataFrame()

df['Load_actual']=df_l['Actual']
df['Load_pred']=df_l['Predicted']
df['Load_base']=df_b['load_pred']

df['PV_actual']=df_pv['Actual']
df['PV_pred']=df_pv['Predicted']
df['PV_base']=df_b['pv_pred']

df['Price_actual']=df_pr['Actual_diff']
df['Price_pred']=df_pr['Predicted_diff']
df['Price_base']=df_b['price_pred']

df['idx']=df_l['final_idx']
df.set_index('idx')
df.drop('idx',axis=1,inplace=True)