import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import solve_ivp
import time

h = 0.01 #step size
t = np.arange(0, 5 + h, h) #time values

#amount of drug in gi and body
def adrug(t, gi, b, ka, k):
    return -ka * gi, ka * gi - k * b

vectorized_adrug = np.vectorize(adrug)

d = 100 #mg
ka = np.random.uniform(low=1,high=10,size=10) #absorption constant
gi0 = d #initial condition
b0 = 0 #initial condition
k = np.random.uniform(low=1,high=10,size=10) #elimination constant

gb = np.zeros((len(t), 2, len(ka)))
gb[0, 0, :] = gi0
gb[0, 1, :] = b0

for i in range(0, len(t) - 1):
    agi, ab = vectorized_adrug(t[i], gb[i, 0, :], gb[i, 1, :], ka, k)
    gb[i + 1, 0, :] = gb[i, 0, :] + h * agi
    gb[i + 1, 1, :] = gb[i, 1, :] + h * ab

gi = gb[:, 0, :]
avg_gi = np.mean(gi, axis=1)
b = gb[:, 1, :]
avg_b = np.mean(b, axis=1)

def adrug_scipy(t, gb, ka, k):
    n = 10
    gi = gb[:n]
    b = gb[n:]
    agi, ab = vectorized_adrug(t, gi, b, ka, k)
    return np.concatenate((agi, ab))

scipy_solved = solve_ivp(adrug_scipy, (0, 5), np.concatenate((np.full(10, gi0), np.full(10, b0))), args=(ka, k), t_eval=t)

solved_gi = scipy_solved.y[:10, :]
avg_solved_gi = np.mean(solved_gi, axis=0)
solved_b = scipy_solved.y[10:, :]
avg_solved_b = np.mean(solved_b, axis=0)

plt.figure(figsize = (12, 8))

plt.subplot(121)
plt.plot(t, avg_gi, 'b--', label="Average GI - Euler's")
plt.plot(t, avg_b, 'r--', label="Average Body - Euler's")
plt.title("Amount of Drug Over Time - Euler's")
plt.xlabel('Time')
plt.ylabel('Amount of Drug (mg)')
plt.legend()
plt.grid()

plt.subplot(122)
plt.plot(t, avg_solved_gi, 'b', label="Average GI - SciPy Integrate")
plt.plot(t, avg_solved_b, 'r', label="Average Body - SciPy Integrate")
plt.title('Amount of Drug Over Time - SciPy Integrate')
plt.xlabel('Time')
plt.ylabel('Amount of Drug (mg)')
plt.legend()
plt.grid()

plt.show()
