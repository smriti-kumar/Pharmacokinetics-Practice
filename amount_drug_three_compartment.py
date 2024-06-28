import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import solve_ivp
from abstract_base_class_models import ThreeCompartmentModel
from helper_methods import euler_func

h = 0.01 #step size
total_time = 5 #total time
t = np.arange(0, total_time + h, h) #time values

N = int(1000)
ka = np.random.uniform(low=1,high=10,size=N) #absorption constant
k10 = np.random.uniform(low=1,high=10,size=N) #elimination constant
k12 = np.random.uniform(low=1,high=10,size=N) #central to peripheral constant
k21 = np.random.uniform(low=1,high=10,size=N) #peripheral to central constant
k13 = np.random.uniform(low=1,high=10,size=N) #central to deep tissue constant
k31 = np.random.uniform(low=1,high=10,size=N) #deep tissue to central constant

gcpd = np.zeros((len(t), 4, N))
inputs = np.zeros_like(t)
dosing_interval = np.argwhere(t % (100 * h) == 0)
for i in range(len(dosing_interval)):
    dosing_interval[i] += int(np.random.rand() * 30 - 15)
    if dosing_interval[i] > 500:
        dosing_interval[i] = 500
    elif dosing_interval[i] < 0:
        dosing_interval[i] = 0
inputs[dosing_interval] = 100 / h 
parameters = np.column_stack((ka, k10, k12, k21, k13, k31)).T

threecomp = ThreeCompartmentModel(gcpd, parameters, inputs, 4)

for i in range(0, len(t) - 1):
    gcpd[i+1] = euler_func(gcpd[i], threecomp.ab(i).T, h)

gi = gcpd[:, 0, :]
avg_gi = np.mean(gi, axis=1)
c = gcpd[:, 1, :]
avg_c = np.mean(c, axis=1)
p = gcpd[:, 2, :]
avg_p = np.mean(p, axis=1)
dt = gcpd[:, 3, :]
avg_dt = np.mean(dt, axis=1)

# def ab_scipy(t, gcpd, ka, k10, k12, k21, k13, k31):
#     gi = gcpd[0:10]
#     c = gcpd[10:20]
#     p = gcpd[20:30]
#     dt = gcpd[30:40]
#     agi, ac, ap, adt = vectorized_ab(t, gi, c, p, dt, ka, k10, k12, k21, k13, k31)
#     return np.concatenate((agi, ac, ap, adt))
    
# scipy_solved = solve_ivp(ab_scipy, (0, 5), np.concatenate((np.full(10, gi0), np.full(10, c0), np.full(10, p0), np.full(10, dt0))), args=(ka, k10, k12, k21, k13, k31), t_eval=t)

# solved_gi = scipy_solved.y[0:10, :]
# avg_solved_gi = np.mean(solved_gi, axis=0)
# solved_c = scipy_solved.y[10:20, :]
# avg_solved_c = np.mean(solved_c, axis=0)
# solved_p = scipy_solved.y[20:30, :]
# avg_solved_p = np.mean(solved_p, axis=0)
# solved_dt = scipy_solved.y[30:40, :]
# avg_solved_dt = np.mean(solved_dt, axis=0)

plt.figure(figsize = (12, 8))

# plt.subplot(121)
plt.plot(t, avg_gi, 'b--', label="Average GI - Euler's")
plt.plot(t, avg_c, 'r--', label="Average Central - Euler's")
plt.plot(t, avg_p, 'g--', label="Average Peripheral - Euler's")
plt.plot(t, avg_dt, 'm--', label="Average Deep Tissue - Euler's")
plt.title("Amount of Drug Over Time - Euler's")
plt.xlabel('Time')
plt.ylabel('Amount of Drug (mg)')
plt.legend()
plt.grid()

# plt.subplot(122)
# plt.plot(t, avg_solved_gi, 'b', label='Average GI - SciPy Integrate')
# plt.plot(t, avg_solved_c, 'r', label='Average Central - SciPy Integrate')
# plt.plot(t, avg_solved_p, 'g', label='Average Peripheral - SciPy Integrate')
# plt.plot(t, avg_solved_dt, 'm', label='Average Deep Tissue - SciPy Integrate')
# plt.title('Amount of Drug Over Time - SciPy Integrate')
# plt.xlabel('Time')
# plt.ylabel('Amount of Drug (mg)')
# plt.legend()
# plt.grid()

plt.show()
