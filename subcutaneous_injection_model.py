import numpy as np
import matplotlib.pyplot as plt
from helper_methods import euler_func

def model(states, time, parameters, inputs):
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

h = 0.01 #step size
total_time = 5 #total time
t = np.arange(0, total_time + h, h) #time values

k1 = 2
k2 = 3
k3 = 4
Vd = 5

drug_in_body = np.zeros((len(t), 3))
inputs = np.zeros_like(t)
inputs[0] = 20
parameters = np.column_stack((k1, k2, k3, Vd)).T

for i in range(0, len(t) - 1):
    x1 = drug_in_body[i, 0]
    x2 = drug_in_body[i, 1]
    x3 = drug_in_body[i, 2]
    states = np.column_stack((x1, x2, x3)).T
    dx1dt, dx2dt, dx3dt = model(states, t, parameters, inputs[i])
    drug_in_body[i+1, 0] = drug_in_body[i, 0] + h * dx1dt
    drug_in_body[i+1, 1] = drug_in_body[i, 1] + h * dx2dt
    drug_in_body[i+1, 2] = drug_in_body[i, 2] + h * dx3dt

x1 = drug_in_body[:, 0]
x2 = drug_in_body[:, 1]
x3 = drug_in_body[:, 2]

plt.figure(figsize = (12, 8))

plt.plot(t, x1, 'b--', label="x1")
plt.plot(t, x2, 'r--', label="x2")
plt.plot(t, x3, 'g--', label="x3")
plt.title("Amount of Drug Over Time - Euler's")
plt.xlabel('Time')
plt.ylabel('Amount of Drug (mg)')
plt.legend()
plt.grid()
plt.show()