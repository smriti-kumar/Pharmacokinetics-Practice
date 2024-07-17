import numpy as np
from helper_methods import euler_func
import scipy
import pandas as pd
import matplotlib.pyplot as plt

h = 0.01 #step size

def adrug(states, time, parameters, inputs):
    """
    Computes the rate of change of the state variables for a subcutaneous injection pharmacokinetics model.

    This function models the pharmacokinetics of a drug administered via subcutaneous injection. It calculates the 
    rate of change of drug concentration in different compartments of the body over time.

    Parameters:
    states (array-like): A list or array of state variables [x1, x2, x3].
        - x1 and x2: Drug mass in transit to plasma.
        - x3: Drug concentration in plasma.
    time (float): The current time (not used in this function but typically included for ODE solvers).
    parameters (array-like): A list or array of parameters [k1, k2, k3, V].
        - V: Volume of distribution.
    inputs (float): The input to the system (dose rate), affecting the rate of change of drug amount in the subcutaneous tissue.

    Returns:
    numpy.ndarray: An array containing the rate of change of the state variables [dx1dt, dx2dt, dx3dt].
        - dx1dt: Rate of change of drug amount in the subcutaneous tissue.
        - dx2dt: Rate of change of drug amount in the blood/plasma.
        - dx3dt: Rate of change of drug amount in the target tissue.
    """
    
    # Unpack the states into individual variables
    x1, x2, x3 = states

    # Unpack the parameters
    k1, k2, k3, V = parameters

    # Calculate the rate of change of x1
    # dx1dt represents the change in x1 over time, which is affected by the input and the rate constant k1
    dx1dt = inputs - k1 * x1

    # Calculate the rate of change of x2
    # dx2dt represents the change in x2 over time, which is produced by x1 and consumed by the rate constant k2
    dx2dt = k1 * x1 - k2 * x2

    # Calculate the rate of change of x3
    # dx3dt represents the change in x3 over time, which is produced by x2 (adjusted by volume V) and consumed by the rate constant k3
    dx3dt = k2 * x2 / V - k3 * x3

    # Return the new state as a numpy array
    return np.array([dx1dt, dx2dt, dx3dt])

# for 20 mu g / kg

df = pd.read_csv("liraglutide-20mug-kg.csv")
df = pd.DataFrame.to_numpy(df)
t = df[:, 0]
b = df[:, 1]
inputs = np.zeros_like(t)
inputs[0] = 20

n = 5
location = np.random.rand(4, n) * 100

def objective_function(k1, k2, k3, Vd):
    global n, h, t, b, inputs
    parameters = np.column_stack((k1, k2, k3, Vd)).T
    if (parameters.shape[1] == n):
        b_sim = np.zeros((len(t), 3, n))
        error = np.zeros(n)
        for i in range(0, len(t) - 1):
            x1 = b_sim[i, 0]
            x2 = b_sim[i, 1]
            x3 = b_sim[i, 2]
            states = np.column_stack((x1, x2, x3)).T
            print(states)
            b_sim[i+1] = euler_func(b_sim[i], adrug(states, t, parameters, inputs[i]), h)
        for i in range(n):
            error[i] = np.sum((b - b_sim[:, 2, i])**2)
    else:
        b_sim = np.zeros((len(t), 3))
        for i in range(0, len(t) - 1):
            x1 = b_sim[i, 0]
            x2 = b_sim[i, 1]
            x3 = b_sim[i, 2]
            states = np.column_stack((x1, x2, x3)).T
            dx1dt, dx2dt, dx3dt = adrug(states, t, parameters, inputs[i])
            b_sim[i+1, 0] = b_sim[i, 0] + h * dx1dt
            b_sim[i+1, 1] = b_sim[i, 1] + h * dx2dt
            b_sim[i+1, 2] = b_sim[i, 2] + h * dx3dt
        error = np.sum((b - b_sim[:, 2])**2)
    return error

func_values = objective_function(location[0], location[1], location[2], location[3])
p_best = func_values
p_best_values = location
g_best = np.min(func_values)
g_best_values = p_best_values.argmin()

w = 0.8
c1 = 0.25
c2 = 0.25

V = np.random.randn(4, n) * 0.1

loop_n = 10
for i in range (loop_n):
    r0 = np.random.rand(1, n)
    r1 = np.random.rand(1, n)
    V = w * V + c1 * r0 * (p_best - location) + c2 * r1 * (g_best - location)
    location = location + V
    func_values = objective_function(location[0], location[1], location[2], location[3])
    p_best_values[:, (p_best >= func_values)] = location[:, (p_best >= func_values)]
    p_best = objective_function(p_best_values[0], p_best_values[1], p_best_values[2], p_best_values[3])
    g_best_values = p_best_values[:, p_best.argmin()]
    g_best = objective_function(g_best_values[0], g_best_values[1], g_best_values[2], g_best_values[3])

k1, k2, k3, Vd = g_best_values

print("k1 =", k1)
print("k2 =", k2)
print("k3 =", k3)
print("Vd =", Vd)

drug_est = np.zeros((len(t), 3))
parameters = np.column_stack((k1, k2, k3, Vd)).T
for i in range(0, len(t) - 1):
    x1 = drug_est[i, 0]
    x2 = drug_est[i, 1]
    x3 = drug_est[i, 2]
    states = np.column_stack((x1, x2, x3)).T
    dx1dt, dx2dt, dx3dt = adrug(states, t, parameters, inputs[i])
    drug_est[i+1, 0] = drug_est[i, 0] + h * dx1dt
    drug_est[i+1, 1] = drug_est[i, 1] + h * dx2dt
    drug_est[i+1, 2] = drug_est[i, 2] + h * dx3dt

plt.plot(t, b, 'b--', label='Given')
plt.plot(t, drug_est[:, 2], 'r--', label='Estimated')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Plasma Concentration of Drug')
plt.title('Model Fitting with PSO')
plt.show()