# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 13:09:07 2022

@author: Karthikeyan
"""


#%% Scalling
#To bring it to 0 to 1 so that measured eualidian distances are uniform
#Check coefficient of variation- std/mean - takes into account of the outliers
#Outliers needs to be taken into considereation i.e std is high/data doesnt have a gaussian dist
# so std is high?--> COV 


df_scale=pd.DataFrame()

#why do we always have to use training data for fitting
scaler1=RobustScaler()
scaler1=scaler1.fit(train)
temp_scale1=np.reshape(np.asarray(train['grid_import']),(len(train),1))
df_scale['train1']=scaler1.transform(temp_scale1).flatten()



scaler2=MinMaxScaler()
scaler2=scaler2.fit(train)
temp_scale2=np.reshape(np.asarray(train['grid_import']),(len(train),1))
df_scale['train2']=scaler2.transform(temp_scale2).flatten()
# scaler2=MinMaxScaler()
# scaler2=scaler2.fit(train)
#df_scale['train2']=scaler2.transform(train)