import numpy as np
from helper_methods import euler_func
import scipy
import pandas as pd
import matplotlib.pyplot as plt

h = 0.01

def adrug(states, time, parameters, inputs):
    x1, x2, x3 = states
    k1, k2, k3, V = parameters
    dx1dt = inputs - k1 * x1
    dx2dt = k1 * x1 - k2 * x2
    dx3dt = k2 * x2 / V - k3 * x3
    return np.array([dx1dt, dx2dt, dx3dt])

def func(t, k1, k2, k3, Vd):
    global inputs, h
    drug_in_body = np.zeros((8, len(t), 3))
    parameters = np.column_stack((k1, k2, k3, Vd)).T
    for j in range(0, 8):
        for i in range(0, len(t) - 1):
            x1 = drug_in_body[j, i, 0]
            x2 = drug_in_body[j, i, 1]
            x3 = drug_in_body[j, i, 2]
            states = np.column_stack((x1, x2, x3)).T
            dx1dt, dx2dt, dx3dt = adrug(states, t, parameters, inputs[i, j])
            drug_in_body[j, i+1, 0] = drug_in_body[j, i, 0] + h * dx1dt
            drug_in_body[j, i+1, 1] = drug_in_body[j, i, 1] + h * dx2dt
            drug_in_body[j, i+1, 2] = drug_in_body[j, i, 2] + h * dx3dt
    return drug_in_body[:, :, 2].flatten()

df = pd.read_csv("GLP1-RA Digitized Data - plasma concentration.csv")
df = pd.DataFrame.to_numpy(df)
t = df[:, 0]
b = df[:, 1:]
inputs = np.zeros((len(t), 8))
inputs[0, :] = [1.25, 2.5, 5, 10, 12.5, 15, 17.5, 20]

parameters, parameters_covariance = scipy.optimize.curve_fit(func, t, b.T.flatten())
k1, k2, k3, Vd = parameters
print("k1 =", k1)
print("k2 =", k2)
print("k3 =", k3)
print("Vd =", Vd)

drug_est = np.zeros((8, len(t), 3))
parameters = np.column_stack((k1, k2, k3, Vd)).T
for j in range(0, 8):
    for i in range(0, len(t) - 1):
        x1 = drug_est[j, i, 0]
        x2 = drug_est[j, i, 1]
        x3 = drug_est[j, i, 2]
        states = np.column_stack((x1, x2, x3)).T
        dx1dt, dx2dt, dx3dt = adrug(states, t, parameters, inputs[i, j])
        drug_est[j, i+1, 0] = drug_est[j, i, 0] + h * dx1dt
        drug_est[j, i+1, 1] = drug_est[j, i, 1] + h * dx2dt
        drug_est[j, i+1, 2] = drug_est[j, i, 2] + h * dx3dt
plt.plot(t, b[:, 0], 'b--', label='Given')
plt.plot(t, b, 'b--')
plt.plot(t, drug_est[0, :, 2].T, 'r--', label='Estimated')
plt.plot(t, drug_est[1:, :, 2].T, 'r--')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Plasma Concentration of Drug')
plt.title('Model Fitting with Scipy')
plt.show()