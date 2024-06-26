import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import solve_ivp

h = 0.01 #step size
t = np.arange(0, 5 + h, h) #time values

#amount of drug in gi and two compartments
def ab (t, gi, c, p, ka, k10, k12, k21):
    return - 1 * ka * gi, ka * gi + k21 * p - k10 * c - k12 * c, k12 * c - k21 * p

vectorized_ab = np.vectorize(ab)

d = 100 #mg
gi0 = d #initial condition for gi
c0 = 0 #initial condition for central
p0 = 0 #initial condition for peripheral
ka = np.random.uniform(low=1,high=10,size=10) #absorption constant
k10 = np.random.uniform(low=1,high=10,size=10) #elimination constant
k12 = np.random.uniform(low=1,high=10,size=10) #central to peripheral constant
k21 = np.random.uniform(low=1,high=10,size=10) #peripheral to central constant

gcp = np.zeros((len(t), 3, len(ka)))
gcp[0, 0, :] = gi0
gcp[0, 1, :] = c0
gcp[0, 2, :] = p0

for i in range(0, len(t) - 1):
    agi, ac, ap = vectorized_ab(t[1], gcp[i, 0, :], gcp[i, 1, :], gcp[i, 2, :], ka, k10, k12, k21)
    gcp[i + 1, 0, :] = gcp[i, 0, :] + h * agi
    gcp[i + 1, 1, :] = gcp[i, 1, :] + h * ac
    gcp[i + 1, 2, :] = gcp[i, 2, :] + h * ap

gi = gcp[:, 0, :]
avg_gi = np.mean(gi, axis=1)
c = gcp[:, 1, :]
avg_c = np.mean(c, axis=1)
p = gcp[:, 2, :]
avg_p = np.mean(p, axis=1)

def ab_scipy(t, gcp, ka, k10, k12, k21):
    gi = gcp[0:10]
    c = gcp[10:20]
    p = gcp[20:30]
    agi, ac, ap = vectorized_ab(t, gi, c, p, ka, k10, k12, k21)
    return np.concatenate((agi, ac, ap))

scipy_solved = solve_ivp(ab_scipy, (0, 5), np.concatenate((np.full(10, gi0), np.full(10, c0), np.full(10, p0))), args=(ka, k10, k12, k21), t_eval=t)

solved_gi = scipy_solved.y[0:10, :]
avg_solved_gi = np.mean(solved_gi, axis=0)
solved_c = scipy_solved.y[10:20, :]
avg_solved_c = np.mean(solved_c, axis=0)
solved_p = scipy_solved.y[20:30, :]
avg_solved_p = np.mean(solved_p, axis=0)

plt.figure(figsize = (12, 8))

plt.subplot(121)
plt.plot(t, avg_gi, 'b--', label="Average GI - Euler's")
plt.plot(t, avg_c, 'r--', label="Average Central - Euler's")
plt.plot(t, avg_p, 'g--', label="Average Peripheral - Euler's")
plt.title("Amount of Drug Over Time - Euler's")
plt.xlabel('Time')
plt.ylabel('Amount of Drug (mg)')
plt.legend()
plt.grid()

plt.subplot(122)
plt.plot(t, avg_solved_gi, 'b', label='Average GI - SciPy Integrate')
plt.plot(t, avg_solved_c, 'r', label='Average Central - SciPy Integrate')
plt.plot(t, avg_solved_p, 'g', label='Average Peripheral - SciPy Integrate')
plt.title('Amount of Drug Over Time - SciPy Integrate')
plt.xlabel('Time')
plt.ylabel('Amount of Drug (mg)')
plt.legend()
plt.grid()

plt.show()
