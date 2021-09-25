#!/usr/local/bin/python3
import sys
import os

import pandas as pd
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import seaborn as sns

# used for testing
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error, r2_score, log_loss
from sklearn.preprocessing import StandardScaler, scale, MinMaxScaler
from sklearn.model_selection import train_test_split , cross_val_score , GridSearchCV
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, LinearRegression, LogisticRegression

import random
import jax.numpy as jnp
from jax import grad

# Q2 a
def get_gradient(x):
    A = np.array([[1 , 0 , 1 , -1],[-1, 1 ,0, 2],[0, -1, -2, 1]])
    b = np.array([1,2,3])
    # print(A)
    # print(b)
    AtA = np.matmul(np.transpose(A),A)
    AtAx = np.matmul(AtA,x)
    gradient = -np.matmul(np.transpose(A),b) + AtAx
    return gradient

def get_condition(x_curr):
    grad = get_gradient(x_curr)
    return la.norm(grad)

def gradient_descent(start_x, step):
    x_s = []
    condition = get_condition(start_x)
    x_s.append([0,start_x])

    k = 1
    xk = start_x

    while (condition >= 0.001):
        grad = get_gradient(xk)
        x_new = xk - step * grad
        x_s.append([k,x_new])

        condition = get_condition(x_new)
        xk = x_new # setting to new values
        k = k+1
        print(k)
    return x_s

start_x = np.array([1,1,1,1])
results = gradient_descent(start_x , 0.1)
print("====first 5========")
for num in results[:5]:
     print(f"k = {num[0]} --- x(k) = {num[1]}")

print("====last 5========")
for num in results[-5:]:
     print(f"k = {num[0]} --- x(k) = {num[1]}")


# Q2 b
def get_gradient(x):
    A = np.array([[1 , 0 , 1 , -1],[-1, 1 ,0, 2],[0, -1, -2, 1]])
    b = np.array([1,2,3])
    # print(A)
    # print(b)
    AtA = np.matmul(np.transpose(A),A)
    AtAx = np.matmul(AtA,x)
    gradient = -np.matmul(np.transpose(A),b) + AtAx
    return gradient

steps = []
def get_step(grad_at_x, x_curr):
    A = np.array([[1 , 0 , 1 , -1],[-1, 1 ,0, 2],[0, -1, -2, 1]])
    b = np.array([1,2,3])

    term1 = -2*np.matmul(np.transpose(b),A)
    term2 = np.matmul(term1,grad_at_x)

    term3 = np.matmul(np.transpose(x_curr),np.transpose(A))
    term4 = np.matmul(term3, A)
    term5 = np.matmul(term4, grad_at_x)

    term6 = np.matmul(np.transpose(grad_at_x), np.transpose(A))
    term7 = np.matmul(term6, A)
    term8 = np.matmul(term7 , x_curr)

    numerator = term2 + term5 + term8

    term9 = 2*np.matmul(np.transpose(grad_at_x),np.transpose(A))
    term10 = np.matmul(term9,A)
    term11 = np.matmul(term10,grad_at_x)

    demoninator = term11

    step = numerator/demoninator
    steps.append(step)
    return step

def get_condition(x_curr):
    grad = get_gradient(x_curr)
    return la.norm(grad)

def gradient_descent(start_x, step):
    x_s = []
    condition = get_condition(start_x)
    x_s.append([0,start_x])

    k = 1
    xk = start_x

    while (condition >= 0.001):
        grad = get_gradient(xk)
        step = get_step(grad, xk)
        x_new = xk - step * grad
        x_s.append([k,x_new])

        condition = get_condition(x_new)
        xk = x_new # setting to new values

        print(k)
        k = k+1
    return x_s

start_x = np.array([1,1,1,1])
results = gradient_descent(start_x , 0.1)
print("====first 5========")
for num in results[:5]:
     print(f"k = {num[0]} --- x(k) = {num[1]}")

print("====last 5========")
for num in results[-5:]:
     print(f"k = {num[0]} --- x(k) = {num[1]}")

sns.scatterplot(data = steps)
plt.title('iterations', fontsize=20)
plt.xlabel('iterations')
plt.ylabel('Step size')
plt.show()

#Q2b - it's hard to say, it could be either from what you've described. One simple way to debug here would be to use scipy.minimize to compute the alphas here to check that your alpha solution is correct, the answers won't be exactly the same but it could be a good guide




#Q2d

df = pd.read_csv("Q2.csv")
df = df.dropna()
x_df = df.drop(columns=['transactiondate','latitude','longitude','price'])
y_df = df[["price"]]
scaler = MinMaxScaler()
scaled_x_df = scaler.fit_transform(x_df)

print(scaled_x_df)

scaled_x_df = pd.DataFrame(data=scaled_x_df)
print(scaled_x_df)
scaled_x_df["1"] = 1
first_col = scaled_x_df.pop("1")
scaled_x_df.insert(0, "1", first_col)

print(scaled_x_df)

x_train, x_test, y_train, y_test = train_test_split(scaled_x_df, y_df, test_size=0.5, shuffle=False)
print(x_train)

y_train = np.asarray(y_train)
y_test = np.asarray(y_test)
x_train = np.asarray(x_train)
x_test = np.asarray(x_test)
print(f"x trian first row  {x_train[0]}")
print(f"x trian last row  {x_train[-1]}")
print(f"x test first 1 {x_test[0]}")
print(f"x test last row 1 {x_test[-1]}")

print(f"y trian first row  {y_train[0]}")
print(f"y trian last row  {y_train[-1]}")
print(f"y test first 1 {y_train[0]}")
print(f"y test last row 1 {y_train[-1]}")

# Q2e
def get_loss(x,y,w):
    WtX = jnp.matmul(w, jnp.transpose(x))
    sqrt_loss = 0.25*((y-WtX)**2)
    sqrt_loss += 1
    loss = jnp.sqrt(sqrt_loss)
    loss -= 1

    return jnp.sum(loss) # changes the shape of the array its just 1 number still

def get_loss_sum(x,y,w):
    k = x.shape[0]
    losses = []
    res = []
    for i in range(k):
        loss = get_loss(x[i],y[i],w)
        losses.append(loss)
    res = jnp.mean(jnp.array(losses))
    return res

def get_gradient(x,y,w):
    k = x.shape[0]
    losses = []
    res = []
    for i in range(k):
        loss = grad(get_loss)(x[i],y[i],w)
        losses.append(loss)
    res = jnp.mean(jnp.array(losses))
    return res
loss_all = []
def gradient_descent_train(w, x, y, step=1,termination_val=0.0001):
    results = {}

    converged = False
    wk = w
    k=0
    inital_loss = get_loss_sum(x, y, wk)
    cond = inital_loss
    while cond >= 0.0001:
        grad = get_gradient(x, y, wk)
        wk_new = wk - step * grad
        results[k] = wk_new

        cond = abs(get_loss_sum(x, y, wk_new) - get_loss_sum(x, y, wk))
        loss_for_wk = get_loss_sum(x, y, wk)
        numpy_array = float(np.asarray(loss_for_wk))

        loss_all.append(numpy_array)
        print("Iteration: ", k)
        #jnp mean is just changing shape of result here
        print("weight_diff: ", jnp.mean(cond))
        wk = wk_new
        k+=1
    return results, wk

w0 = np.array([1,1,1,1])
initial_step = 1
res, final_w = gradient_descent_train(w0, x_train, y_train, initial_step)
print(final_w)
print("-------loss all ")
print(loss_all)

#getting train loss
train_loss = float(np.asarray(get_loss_sum(x_train, y_train, final_w)))
test_loss = float(np.asarray(get_loss_sum(x_test, y_test, final_w)))

print(f"train_loss {train_loss}")
print(f"test_loss {test_loss}")



sns.scatterplot(data = loss_all)
plt.title('training loss at each step of the algorithm', fontsize=10)
plt.xlabel('iterations')
plt.ylabel('loss')
plt.show()




#Q2f)




























        #
