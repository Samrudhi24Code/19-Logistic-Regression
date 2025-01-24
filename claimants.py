# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 15:17:55 2025

@author: Dell
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as sm
#above pkg is used in the multiple linear regression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import classification_report

claimants=pd.read_csv("E:/Honars(DS)/Data Science/19-Logistic Regression/claimants.csv")
#there are CLAMAGE abd LOSS are having continuous data rest are verify the 
#dataset,where the CASENUM is not really useful 
c1=claimants.drop('CASENUM',axis=1)
c1.head()
c1.describe()
#let us check there null values
c1.isna().sum()
c1.info()

#there are sevral null values 
#if we will used dropna() function we will lose 290 data points 
#hence we will go for imputation ]
#if the data is categorical then we have to fill the null valies with mean imputation
c1.dtypes
mean_value=c1.CLMAGE.mean()
mean_value
#now let us impute the same
c1.CLMAGE=c1.CLMAGE.fillna(mean_value)
c1.CLMAGE.isna().sum()


#hence all the null values of clmage has been filled by mean value
#for columns where there are decsecrete values we will apply mode
mode_CLMSEX=c1.CLMSEX.mode()
mode_CLMSEX
c1.CLMSEX=c1.CLMSEX.fillna((mode_CLMSEX)[0])
c1.CLMSEX.isna().sum()


#CLMINSUR is also categorical data hence mode imputation is applied
# Impute missing values for CLMINSUR
mode_CLMINSUR = c1.CLMINSUR.mode()[0]  # Ensure accessing the first mode value
c1.CLMINSUR = c1.CLMINSUR.fillna(mode_CLMINSUR)

# Verify the imputation
print(c1.CLMINSUR.isna().sum())

# Double-check SEATBELT


#SEATBELT is categorical data hence go for mode imputation
mode_SEATBELT=c1.SEATBELT.mode()
mode_SEATBELT
c1.SEATBELT=c1.SEATBELT.fillna((mode_SEATBELT)[0])
c1.SEATBELT.isna().sum()


#now the personwe met an accident will hire the attenrev or not
##let us build the model
import statsmodels.api as sm

# Fit the logistic regression model
logit_model = sm.Logit.from_formula('ATTORNEY ~ CLMAGE + LOSS + CLMINSUR + CLMSEX + SEATBELT', data=c1).fit()

# Display the summary of the model
print(logit_model.summary())
print(logit_model.summary2())
#in logistic regression we do not have R squared values,onlyb check p=values
##SEATBELT is staistically insignificant ignore and proceed

#her we are going to check AIC value,it stands for akike information 
#is mathematical method for the evaluation how well a model fits the data
#a lower the score the better model

#now let us check for predictions
pred=logit_model.predict(c1.iloc[:,1:])
#here we are applying all the rows columns from 1 as columns 0 is ATTORNEy
#target value

#to derive the ROC curve 
#Roc curve has tpr on y-axis and fpr on x-axis, ideally tpr must be high
#and fpr must be low
fpr,tpr,thresholds=roc_curve(c1.ATTORNEY,pred)
#to identify the optimum thresholds
print(c1.isna().sum())

optimal_idx=np.argmax(tpr-fpr)
optimal_threshold=thresholds[optimal_idx]
optimal_threshold

#0.52944,by default you can take 0.5 valye as a threshold Now we want to  identify if new value is given
# to the modelit willfall in which the region of 0 or 1 

import pylab as pl
i=np.arange(len(tpr))
roc=pd.DataFrame({
    'fpr':pd.Series(fpr,index=i),
    'tpr':pd.Series(tpr,index=i),
    '1-fpr':pd.Series(1-fpr,index=i),
    'tf':pd.Series(tpr-(1-fpr),index=i),
    'thresholds':pd.Series(thresholds,index=i)
    })

plt.plot(fpr,tpr)
plt.xlabel("FPR");plt.ylabel("TPR")
roc_auc=auc(fpr,tpr)
print("Area Under the curve is",roc_auc)

#now let us  add the prediction column in the Data Frame

c1['pred']=np.zeros(1340)
c1.loc[pred>optimal_threshold,'pred']=1

#if the predicted value is greater than the optimal threshold
classification=classification_report(c1['pred'],c1['ATTORNEY'])
classification

#splitting the data in treain test  plit

train_data,test_data=train_test_split(c1,test_size=0.3)
model = sm.Logit.from_formula('ATTORNEY ~ CLMAGE + LOSS + CLMINSUR + CLMSEX + SEATBELT', data=train_data).fit()
print(model.summary())
print(model.summary2())
#ATC is 1168.

test_pred=model.predict(test_data)
test_data['test_pred']=np.zeros(402)
test_data.loc[test_pred>optimal_threshold,'test_pred']=1
#confusion_matrix

confusion_matrix=pd.crosstab(test_data.test_pred,test_data.ATTORNEY)
confusion_matrix
acc_test=(310+343)/938
acc_test
#0.726

#classification_report

class_report = classification_report(test_data['ATTORNEY'], test_data['test_pred'])
class_report

#Roc AUC curve
fpr,tpr,thresholds=roc_curve(test_data.ATTORNEY,test_pred)

plt.plot(fpr,tpr)
plt.xlabel("FPR");plt.ylabel("TPR")
roc_auc_test=metrics.auc(fpr,tpr)
roc_auc_test

#Prediction on the train_data
train_pred=model.predict(train_data.iloc[:,1:])
#creating new column
train_data["train_pred"]=np.zeros(938)
train_data.loc[train_pred>optimal_threshold,'train_pred']=1
#cofusion matrix

confusion_matrix=pd.crosstab(train_data.train_pred,train_data.ATTORNEY)
confusion_matrix

##############################################################
accuracy_train=(310+342)/652
accuracy_train

#classification_report

class_train=classification_report(train_data.train_pred,train_data.ATTORNEY)
class_train

# ROC_AUC curve
roc_auc_train = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.show()




