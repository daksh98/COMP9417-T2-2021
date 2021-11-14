#!/usr/local/bin/python3
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# used for testing
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, scale
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, LinearRegression



df=pd.read_csv('data.csv')
print(f"{df}")
target_name="Y"
target=df[target_name]

# Q2
#part a)
sns.set_style("whitegrid")
sns.pairplot(df)
plt.show()




# part b
print(df.info())
# standardized_dataset = (dataset - mean(dataset)) / standard_deviation(dataset))
normalized_df = (df-df.mean())/(df.agg(np.std, ddof=0)) #ddofint, default 1 ... Delta Degrees of Freedom. The divisor used in calculations is N - ddof, where N represents the number of elements.
normalized_df["Y"]=df["Y"]
print(df)
print(normalized_df)
print("----mean----")
print(normalized_df.mean())
print("----variance----")
print(normalized_df.agg(np.std, ddof=0))
print("----sum of sq's----")

#checking sum of squares
sum_sqaures = normalized_df['X1'].pow(2)
sum_sqaures = sum_sqaures.sum()
print(f"X1 - {sum_sqaures}")
sum_sqaures = normalized_df['X2'].pow(2)
sum_sqaures = sum_sqaures.sum()
print(f"X2 - {sum_sqaures}")
sum_sqaures = normalized_df['X3'].pow(2)
sum_sqaures = sum_sqaures.sum()
print(f"X3 - {sum_sqaures}")
sum_sqaures = normalized_df['X4'].pow(2)
sum_sqaures = sum_sqaures.sum()
print(f"X4 - {sum_sqaures}")
sum_sqaures = normalized_df['X5'].pow(2)
sum_sqaures = sum_sqaures.sum()
print(f"X5 - {sum_sqaures}")
sum_sqaures = normalized_df['X6'].pow(2)
sum_sqaures = sum_sqaures.sum()
print(f"X6 - {sum_sqaures}")
sum_sqaures = normalized_df['X7'].pow(2)
sum_sqaures = sum_sqaures.sum()
print(f"X7 - {sum_sqaures}")
sum_sqaures = normalized_df['X8'].pow(2)
sum_sqaures = sum_sqaures.sum()
print(f"X8 - {sum_sqaures}")



# part c - Ridge regression Plots
# summarize shape
print(normalized_df.shape)
y = normalized_df['Y']
X = normalized_df.drop('Y',axis=1)

# get our coefficents for each lambda
lambdas = np.array([0.01, 0.1, 0.5, 1, 1.5, 2, 5, 10, 20, 30, 50, 100, 200, 300])
ridge = Ridge()
coefs = []
for a in lambdas:
    ridge.set_params(alpha = a)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)

print(coefs)

# plot
ax = plt.gca()
labels = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']
color_cycle= ['red', 'brown', 'green', 'blue', 'orange', 'pink', 'purple', 'grey']
for i in range(len(coefs[0])):
    ax.plot(lambdas,[pt[i] for pt in coefs], color=color_cycle[i],label = f"{labels[i]}")

ax.set_xscale('log')
ax.autoscale()
plt.xlabel('lambdas')
plt.ylabel('coefficents')
plt.title('Ridge Coefficents V lambdas')
plt.legend()
plt.show()



#part d LOOCV from scratch
#find optimal lambda
lambdas = np.arange(start=0, stop=50.1, step=0.1)
copy_X = normalized_df.drop('Y',axis=1)
copy_y = normalized_df['Y']

ridge_d = Ridge()
row_it = 0
mse_for_lambdas = []

for a in lambdas:
    sq_errors = []
    while row_it < 38:  # Note : one observation is equal to one row per defintion of MLR
        dropped_row_X = copy_X.loc[row_it]
        dropped_y = copy_y.iloc[row_it]

        #remove the ith obs
        ith_removed_X = copy_X.drop(row_it, axis = 0)
        ith_removed_y = copy_y.drop(row_it, axis = 0)
        ridge_d.set_params(alpha = a) # set to current lambda
        ridge_d.fit(ith_removed_X, ith_removed_y)
        dropped_row_X = dropped_row_X.values.reshape(1, -1)

        #get prediction error
        prediction = ridge_d.predict(dropped_row_X)
        prediction_error = pow((dropped_y - prediction[0]),2)
        sq_errors.append(prediction_error)
        row_it= row_it + 1

    row_it = 0
    mse = (sum(sq_errors))/38 # find the Leave one out error average for lambda = a
    mse_for_lambdas.append(mse)

#find best lambda
minMSE = min(mse_for_lambdas)
minMSE_index = mse_for_lambdas.index(min(mse_for_lambdas))
best_lambda = lambdas[minMSE_index]
print(f"MIN mse =  {minMSE} --------------------------")
print(f"best lambda =  {best_lambda} --------------------------")

#plot
ax = plt.gca()
ax.set_xscale('log')
ax.autoscale()
plt.xlabel('lambdas')
plt.ylabel('Leave-one-out-error')
plt.title('Lambdas V LOOE - RidgeCV')
plt.plot(lambdas, mse_for_lambdas, color ="tab:blue")
plt.show()

ols = LinearRegression()
sq_errors = []
# LOOCV for OLS
while row_it < 38:
    dropped_row_X = copy_X.loc[row_it]
    dropped_y = copy_y.iloc[row_it]
    ith_removed_X = copy_X.drop(row_it, axis = 0)
    ith_removed_y = copy_y.drop(row_it, axis = 0)

    ols.fit(ith_removed_X, ith_removed_y)
    dropped_row_X = dropped_row_X.values.reshape(1, -1)

    prediction = ols.predict(dropped_row_X)
    prediction_error = pow((dropped_y - prediction[0]),2)
    sq_errors.append(prediction_error)
    row_it= row_it + 1

mse = (sum(sq_errors))/38
print(f"the LOOE from OLS is {mse}")



#Q2 part e
#get our lambdas
lambdas = np.array([0.01, 0.1, 0.5, 1, 1.5, 2, 5, 10, 20, 30, 50, 100, 200, 300])

copy_X = normalized_df.drop('Y',axis=1)
copy_y = normalized_df['Y']
lasso = Lasso()
coefs = []

for a in lambdas:
    lasso.set_params(alpha = a)
    lasso.fit(copy_X, copy_y)
    coefs.append(lasso.coef_)
#sanity check
print(coefs)
#plot
ax = plt.gca()
labels = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']
# ax.plot(lambdas, coefs)
color_cycle= ['red', 'brown', 'green', 'blue', 'orange', 'pink', 'purple', 'grey']
for i in range(len(coefs[0])):
    ax.plot(lambdas,[pt[i] for pt in coefs], color=color_cycle[i],label = f"{labels[i]}")

ax.set_xscale('log')
ax.autoscale()
plt.axis('tight')
plt.xlabel('lambdas')
plt.ylabel('coefficents')
plt.title('Lasso Coefficents V lambdas')
plt.legend()
plt.show()



#Q2 part f
lambdas = np.arange(start=0, stop=20.1, step=0.1)
copy_X = normalized_df.drop('Y',axis=1)
copy_y = normalized_df['Y']

lasso = Lasso()
row_it = 0
mse_for_lambdas = []

for a in lambdas:
    sq_errors = []

    while row_it < 38:
        dropped_row_X = copy_X.loc[row_it]
        dropped_y = copy_y.iloc[row_it]
        #remove the ith obs
        ith_removed_X = copy_X.drop(row_it, axis = 0)
        ith_removed_y = copy_y.drop(row_it, axis = 0)

        #Predict
        lasso.set_params(alpha = a) # set to current lambda
        lasso.fit(ith_removed_X, ith_removed_y)
        dropped_row_X = dropped_row_X.values.reshape(1, -1)
        prediction = lasso.predict(dropped_row_X)
        prediction_error = pow((dropped_y - prediction[0]),2)
        sq_errors.append(prediction_error)

        row_it= row_it + 1

    row_it = 0
    mse = (sum(sq_errors))/38
    mse_for_lambdas.append(mse)
# Find minimum LOOE for a lambda
minMSE = min(mse_for_lambdas)
minMSE_index = mse_for_lambdas.index(min(mse_for_lambdas))
best_lambda = lambdas[minMSE_index]
print(f"MIN mse =  {minMSE} --------------------------")
print(f"best lambda =  {best_lambda} --------------------------")
#plot graph
ax = plt.gca()
ax.set_xscale('log')
ax.autoscale()
plt.xlabel('lambdas')
plt.ylabel('Leave-one-out-error')
plt.title('Lambdas V LOOE - LassoCV')
plt.plot(lambdas, mse_for_lambdas, color ="tab:orange")
plt.show()
