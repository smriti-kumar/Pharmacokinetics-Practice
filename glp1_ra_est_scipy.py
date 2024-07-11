import numpy as np
from helper_methods import euler_func
import scipy
import pandas as pd
import matplotlib.pyplot as plt

h = 0.01 #step size

def adrug(t, b, ka, k, inputs, i):
   return ka * inputs[i] - k * b

def func(t, ka, k):
    global inputs, h
    b_new = np.zeros(len(t))
    for i in range(0, len(t) - 1):
        b_new[i + 1] = b_new[i] + h * adrug(t[i], b_new[i], ka, k, inputs, i)
    return b_new.flatten()
"""
# for 10 mu g/ kg

df = pd.read_csv("liraglutide-10mug-kg.csv")
df = pd.DataFrame.to_numpy(df)
t = df[:, 0]
b = df[:, 1]
inputs = np.zeros_like(t)
inputs[0] = 10

parameters, parameters_covariance = scipy.optimize.curve_fit(func, t, b)
ka, k = parameters
print("For 10 mu g / kg")
print("ka =", ka)
print("k =", k)

# for 15 mu g/ kg

df = pd.read_csv("liraglutide-15mug-kg.csv")
df = pd.DataFrame.to_numpy(df)
t = df[:, 0]
b = df[:, 1]
inputs = np.zeros_like(t)
inputs[0] = 15

parameters, parameters_covariance = scipy.optimize.curve_fit(func, t, b)
ka, k = parameters
print("For 15 mu g / kg")
print("ka =", ka)
print("k =", k)
"""
# for 20 mu g / kg

df = pd.read_csv("liraglutide-20mug-kg.csv")
df = pd.DataFrame.to_numpy(df)
t = df[:, 0]
b = df[:, 1]
inputs = np.zeros_like(t)
inputs[0] = 20

parameters, parameters_covariance = scipy.optimize.curve_fit(func, t, b)
ka, k = parameters
print("For 20 mu g / kg")
print("ka =", ka)
print("k =", k)

b_est = np.zeros(len(t))
for i in range(0, len(t) - 1):
    b_est[i + 1] = b_est[i] + h * adrug(t[i], b_est[i], ka, k, inputs, i)

plt.plot(t, b, 'b--', label='Given')
plt.plot(t, b_est, 'r--', label='Estimated')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Amount of Drug (mg)')
plt.title('Model Fitting with Scipy')
plt.show()
