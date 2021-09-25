#!/usr/local/bin/python3
import sys
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

# used for testing
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error, r2_score, log_loss,accuracy_score
from sklearn.preprocessing import StandardScaler, scale
from sklearn.model_selection import train_test_split , cross_val_score , GridSearchCV
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, LinearRegression, LogisticRegression


data = pd.read_csv("Q1.csv")
X = data.iloc[:,:-1]
Y = data.iloc[:,-1]


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.23, train_size=0.77,  shuffle=False, stratify=None)
data_train = pd.concat([X_train, Y_train], axis=1)

print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
#
# #cross validation on training set ...
# def cross_validation_split(dataset, folds):
#         dataset_split = [] # construct an array of arrays
#         df_copy = dataset
#         fold_size = int(df_copy.shape[0] / folds)
#
#         j = 0 # counter for jth row
#         for i in range(folds):
#             fold = []
#             # while loop to add elements to the folds
#             while len(fold) < fold_size:
#                 # save the selected line
#                 fold.append(df_copy.loc[j].values.tolist())
#                 # delete the selected line from dataframe not to select again
#                 df_copy = df_copy.drop(j)
#                 #print(df_copy)
#                 j = j+1
#             # save the fold
#             dataset_split.append(np.asarray(fold))
#         return dataset_split
#
# def cross_validation_split_Y(dataset, folds):
#         dataset_split = []
#         df_copy = dataset
#         fold_size = int(df_copy.shape[0] / folds)
#         j = 0
#         for i in range(folds):
#             fold = []
#             while len(fold) < fold_size:
#                 fold.append(df_copy.loc[j].tolist())
#                 df_copy = df_copy.drop(j)
#                 j = j+1
#             dataset_split.append(np.asarray(fold))
#         return dataset_split
#
# data_k_X=cross_validation_split(X_train,10)
# data_k_Y=cross_validation_split_Y(Y_train,10)
#
# # CV to find best C parameter
# C_s = np.linspace(start=0.0001, stop=0.6, num=100, endpoint=True)
# print(C_s)
#
# logreg = LogisticRegression(penalty='l1', solver='liblinear', random_state=0)
# avg_error_C = []
# log_errors_all = []
#
# i = 0
# for a in C_s:
#     log_errors = []
#     while i < 10:
#         # get x and y for the ith fold
#         fold_X=data_k_X[i]
#         fold_Y=data_k_Y[i]
#         # create a 'ith removed' fold
#         ith_removed_X = np.delete(data_k_X, i, axis=0)# dataset without i-th observation
#         ith_removed_Y = np.delete(data_k_Y, i, axis=0)
#         #shaping
#         new_arr_X = ith_removed_X.reshape(-1, ith_removed_X.shape[-1])
#         ith_removed_X_df = pd.DataFrame(new_arr_X)
#         ith_removed_Y = ith_removed_Y.reshape(-1)
#         #fit model to df with out the ith fold
#         logreg.set_params(C = a) # set to current C
#         logreg.fit(ith_removed_X_df, ith_removed_Y)
#
#         # error based on the dropped fold i.e predict using the dropped fold
#         y_pred_probs_t = logreg.predict_proba(fold_X)
#         loss_test = log_loss(fold_Y, y_pred_probs_t)
#
#         log_errors.append(loss_test)
#         i = i+1
#     i = 0
#     avg = (sum(log_errors))/10 # find the Leave one out error average for lambda = a
#     avg_error_C.append(avg)
#     log_errors_all.append(log_errors)
#
# df = pd.DataFrame(log_errors_all)
#
#box plot creation
df_new = df.T
round_C_s = [round(num, 6) for num in C_s]
df_new.columns = round_C_s
print(df_new)

fig, ax = plt.subplots(figsize=(20,10))
sns.boxplot(data=df_new)
plt.title('Choice of C for each 10 Fold CV', fontsize=20, y=1.02)
plt.xticks(rotation='vertical')
plt.tight_layout
plt.show()
#
# print("avg errors---------------------")
# print(avg_error_C)
#
# # find best C
# minavg = min(avg_error_C)
# minavg_index = avg_error_C.index(min(avg_error_C))
# best_C = C_s[minavg_index]
# print(f"MIN avg =  {minavg} ")
# print(f"best C =  {best_C} ")
# # Re-fit the model with this chosen C, and report both train and test accuracy using this model.
# logreg.set_params(C = best_C)
# logreg.fit(X_train, Y_train)
# #traning error
# score_train = accuracy_score(Y_train, logreg.predict(X_train))
# # test error
# score_test = accuracy_score(Y_test, logreg.predict(X_test))
# print(f"train accuracy  =  {score_train} ")
# print(f"test accuracy  =  {score_test} ")
#
#
#
# #Q1 c)
# parameter_candidates = [
#   {'C': C_s},
# ]
# grid_lr = GridSearchCV(estimator= LogisticRegression(penalty='l1' , solver='liblinear',random_state=0),
#                                         cv=10,scoring='neg_log_loss' ,param_grid=parameter_candidates)
# grid_lr.fit(X_train, Y_train)
# best_parameters = grid_lr.best_params_
# print(f"best para grd search {best_parameters}")
#


#Q1 d)
np.random.seed(12)
i = 0
coeffs = list()
coeffs_avg = list()
logreg = LogisticRegression(penalty='l1', C=1, solver='liblinear', random_state=0)

while i < 10:
    choices_X = list()
    choices_Y = list()
    features = X_train.columns
    response = ['Y']
    choice_indices = np.random.choice(len(data_train), 500, replace=True)
    print(f"-{i}-")

    for j in choice_indices:
        choices_X.append(X_train.loc[j].values.tolist())
        choices_Y.append(Y_train.loc[j].tolist())
    choices_X_df = pd.DataFrame(choices_X, columns=features)
    logreg.fit(choices_X_df, choices_Y)
    # dont need intercept
    temp = logreg.coef_
    coeffs.append(temp[0]) # use .tolist if need be
    #print(coeffs)
    i = i+1

coeff_np = np.array(coeffs)
print("================= mean for each beta")
mean_for_each_beta = np.mean(coeff_np, axis = 0)
print(np.mean(coeff_np, axis = 0))
print("================= lower quantiles for each beta ")
L_for_each_beta = np.quantile(coeff_np, .05, axis =0)
print(L_for_each_beta)
print("================= upper quantiles for each beta")
U_for_each_beta = np.quantile(coeff_np, .95, axis =0)
print(U_for_each_beta)


#plotting
plt.figure(figsize=(10,10))
i = 0
while i < 45:
    if L_for_each_beta[i] <= 0 <= U_for_each_beta[i]:
        plt.vlines(i, L_for_each_beta[i],  U_for_each_beta[i], colors='r', linestyles='solid')
        plt.scatter(i, mean_for_each_beta[i])
    else:
        plt.vlines(i, L_for_each_beta[i],  U_for_each_beta[i], colors='b', linestyles='solid')
        plt.scatter(i, mean_for_each_beta[i])
    i = i+1
plt.xlabel(r"Betas")
plt.ylabel("y - axis")
plt.title("Confidence Intervals for Betas")
plt.show()



# Q1 e)
#
# the data generating distribution refers to the relationship between the response Y and the covariates X
# many of the covraites are unessary as seen from the amount of red bars
#
