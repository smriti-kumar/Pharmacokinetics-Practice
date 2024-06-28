import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import solve_ivp
from abstract_base_class_models import TwoCompartmentModel
from helper_methods import euler_func

h = 0.01 #step size
total_time = 5 #total time
t = np.arange(0, total_time + h, h) #time values

N = int(1000)
ka = np.random.uniform(low=1,high=10,size=N) #absorption constant
k10 = np.random.uniform(low=1,high=10,size=N) #elimination constant
k12 = np.random.uniform(low=1,high=10,size=N) #central to peripheral constant
k21 = np.random.uniform(low=1,high=10,size=N) #peripheral to central constant

gcp = np.zeros((len(t), 3, N))
inputs = np.zeros_like(t)
dosing_interval = np.argwhere(t % (100 * h) == 0)
inputs[dosing_interval] = 100 / h 
parameters = np.column_stack((ka, k10, k12, k21)).T

twocomp = TwoCompartmentModel(gcp, parameters, inputs, 4)

for i in range(0, len(t) - 1):
    gcp[i+1] = euler_func(gcp[i], twocomp.ab(i).T, h)
    
gi = gcp[:, 0, :]
avg_gi = np.mean(gi, axis=1)
c = gcp[:, 1, :]
avg_c = np.mean(c, axis=1)
p = gcp[:, 2, :]
avg_p = np.mean(p, axis=1)

# def ab_scipy(t, gcp, ka, k10, k12, k21):
#     gi = gcp[0:N]
#     c = gcp[N:2 * N]
#     p = gcp[2 * N:3 * N]
#     agi, ac, ap = vectorized_ab(t, gi, c, p, ka, k10, k12, k21)
#     return np.concatenate((agi, ac, ap))

# scipy_solved = solve_ivp(ab_scipy, (0, 5), np.concatenate((np.full(N, gi0), np.full(N, c0), np.full(N, p0))), args=(ka, k10, k12, k21), t_eval=t)

# solved_gi = scipy_solved.y[0:N, :]
# avg_solved_gi = np.mean(solved_gi, axis=0)
# solved_c = scipy_solved.y[N:2 * N, :]
# avg_solved_c = np.mean(solved_c, axis=0)
# solved_p = scipy_solved.y[2 * N:3 * N, :]
# avg_solved_p = np.mean(solved_p, axis=0)

# plt.figure(figsize = (12, 8))

# plt.subplot(121)
plt.plot(t, avg_gi, 'b--', label="Average GI - Euler's")
plt.plot(t, avg_c, 'r--', label="Average Central - Euler's")
plt.plot(t, avg_p, 'g--', label="Average Peripheral - Euler's")
plt.title("Amount of Drug Over Time - Euler's")
plt.xlabel('Time')
plt.ylabel('Amount of Drug (mg)')
plt.legend()
plt.grid()

# plt.subplot(122)
# plt.plot(t, avg_solved_gi, 'b', label='Average GI - SciPy Integrate')
# plt.plot(t, avg_solved_c, 'r', label='Average Central - SciPy Integrate')
# plt.plot(t, avg_solved_p, 'g', label='Average Peripheral - SciPy Integrate')
# plt.title('Amount of Drug Over Time - SciPy Integrate')
# plt.xlabel('Time')
# plt.ylabel('Amount of Drug (mg)')
# plt.legend()
# plt.grid()

plt.show()
