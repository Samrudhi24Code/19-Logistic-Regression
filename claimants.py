# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 15:17:55 2025

@author: Dell
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as sm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, classification_report

# Business Objective: 
# The goal of this analysis is to predict whether an individual will hire an attorney after a car accident
# based on various factors like the claimant's age (CLMAGE), loss amount (LOSS), insurance status (CLMINSUR),
# gender (CLMSEX), and seatbelt usage (SEATBELT).

# Problem Statement: 
# The dataset consists of several features such as the claimant's age, loss, and some categorical information.
# The objective is to build a logistic regression model to predict the binary outcome variable ATTORNEY (1 = hired, 0 = not hired).

# Optimization Goal:
# We aim to minimize errors in predicting whether a claimant hires an attorney or not, based on the given features.

# Load the dataset
claimants = pd.read_csv("E:/Honars(DS)/Data Science/19-Logistic Regression/claimants.csv")

# Data Preprocessing:
# Remove the 'CASENUM' column as it is not relevant for the analysis
c1 = claimants.drop('CASENUM', axis=1)

# Check for null values in the dataset
print(c1.isna().sum())  # Sum of missing values in each column

# Handle missing data:
# - For continuous variables, we will fill missing values with the mean.
# - For categorical variables, we will use the mode (most frequent value).

# Impute missing values for CLMAGE (continuous variable) with the mean value
mean_value = c1.CLMAGE.mean()
c1.CLMAGE = c1.CLMAGE.fillna(mean_value)

# Impute missing values for CLMSEX (categorical variable) with the mode
mode_CLMSEX = c1.CLMSEX.mode()[0]
c1.CLMSEX = c1.CLMSEX.fillna(mode_CLMSEX)

# Impute missing values for CLMINSUR (categorical variable) with the mode
mode_CLMINSUR = c1.CLMINSUR.mode()[0]
c1.CLMINSUR = c1.CLMINSUR.fillna(mode_CLMINSUR)

# Impute missing values for SEATBELT (categorical variable) with the mode
mode_SEATBELT = c1.SEATBELT.mode()[0]
c1.SEATBELT = c1.SEATBELT.fillna(mode_SEATBELT)

# Logistic Regression Model:
# Build a logistic regression model to predict the ATTORNEY column based on the features
import statsmodels.api as sm

# Fit the logistic regression model using a formula
logit_model = sm.Logit.from_formula('ATTORNEY ~ CLMAGE + LOSS + CLMINSUR + CLMSEX + SEATBELT', data=c1).fit()

# Display the summary of the model to understand the coefficients and statistical significance
print(logit_model.summary())
print(logit_model.summary2())

# In logistic regression, we do not get R-squared values, so we rely on p-values to determine statistical significance.
# SEATBELT is found to be statistically insignificant, so we can ignore it in further analysis.

# Model Evaluation using AIC (Akaike Information Criterion):
# A lower AIC value indicates a better fit for the model. 
# This step helps in model comparison, where we compare the current model with others.

# Make predictions using the fitted logistic regression model
pred = logit_model.predict(c1.iloc[:, 1:])  # Exclude the target variable ATTORNEY from the predictors

# ROC Curve Analysis:
# We use the Receiver Operating Characteristic (ROC) curve to evaluate the performance of the model.
# The ROC curve plots the True Positive Rate (TPR) vs False Positive Rate (FPR), and we aim to maximize TPR while minimizing FPR.

fpr, tpr, thresholds = roc_curve(c1.ATTORNEY, pred)

# Find the optimal threshold by maximizing TPR - FPR
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print("Optimal Threshold: ", optimal_threshold)

# Plot the ROC curve
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")

# Calculate the area under the curve (AUC)
roc_auc = auc(fpr, tpr)
print("Area Under the Curve (AUC):", roc_auc)

# Add the prediction column to the dataset based on the optimal threshold
c1['pred'] = np.zeros(1340)
c1.loc[pred > optimal_threshold, 'pred'] = 1

# Evaluate the classification performance using classification report
classification = classification_report(c1['pred'], c1['ATTORNEY'])
print("Classification Report:\n", classification)

# Train-Test Split:
# Split the data into training and testing sets (70% for training, 30% for testing)
train_data, test_data = train_test_split(c1, test_size=0.3)

# Train the logistic regression model using the training data
model = sm.Logit.from_formula('ATTORNEY ~ CLMAGE + LOSS + CLMINSUR + CLMSEX + SEATBELT', data=train_data).fit()

# Display the summary of the model to assess the performance on the training set
print(model.summary())
print(model.summary2())

# Make predictions on the test set
test_pred = model.predict(test_data)

# Add the prediction column to the test set based on the optimal threshold
test_data['test_pred'] = np.zeros(402)
test_data.loc[test_pred > optimal_threshold, 'test_pred'] = 1

# Evaluate the confusion matrix for the test data
confusion_matrix = pd.crosstab(test_data.test_pred, test_data.ATTORNEY)
print("Confusion Matrix:\n", confusion_matrix)

# Calculate the accuracy of the model on the test data
acc_test = (310 + 343) / 938
print("Test Accuracy: ", acc_test)

# Generate classification report for the test data
class_report = classification_report(test_data['ATTORNEY'], test_data['test_pred'])
print("Test Classification Report:\n", class_report)

# Evaluate the ROC AUC curve for the test data
fpr, tpr, thresholds = roc_curve(test_data.ATTORNEY, test_pred)
roc_auc_test = metrics.auc(fpr, tpr)
print("Test ROC AUC:", roc_auc_test)

# Predictions on the training data
train_pred = model.predict(train_data.iloc[:, 1:])

# Create a new column in the training data for predictions
train_data["train_pred"] = np.zeros(938)
train_data.loc[train_pred > optimal_threshold, 'train_pred'] = 1

# Evaluate the confusion matrix for the training data
confusion_matrix = pd.crosstab(train_data.train_pred, train_data.ATTORNEY)
print("Training Confusion Matrix:\n", confusion_matrix)

# Calculate the accuracy of the model on the training data
accuracy_train = (310 + 342) / 652
print("Train Accuracy: ", accuracy_train)

# Generate classification report for the training data
class_train = classification_report(train_data.train_pred, train_data.ATTORNEY)
print("Training Classification Report:\n", class_train)

# Plot the ROC AUC curve for the training data
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Training Data")
plt.show()
