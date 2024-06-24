import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import solve_ivp

h = 0.01 #step size
t = np.arange(0, 5 + h, h) #time values

#amount of drug in gi and two compartments
def ab (t, gcp, ka, k10, k12, k21):
    gi, c, p = gcp
    return [- 1 * ka * gi,
            ka * gi + k21 * p - k10 * c - k12 * c,
            k12 * c - k21 * p]

d = 100 #mg
gi0 = d #initial condition for gi
c0 = 0 #initial condition for central
p0 = 0 #initial condition for peripheral
ka = 4 #absorption constant
k10 = 3 #elimination constant
k12 = 2 #central to peripheral constant
k21 = 1 #peripheral to central constant

gcp = np.zeros((len(t), 3))
gcp[0,:] = [gi0, c0, p0]

for i in range(0, len(t) - 1):
    gi, c, p = gcp[i]
    agi, ac, ap = ab(t[i], gcp[i], ka, k10, k12, k21)
    gcp[i + 1] = [gi + h * agi, c + h * ac, p + h * ap]

gi = gcp[:, 0]
c = gcp[:, 1]
p = gcp[:, 2]

scipy_solved = solve_ivp(ab, (0, 5), gcp[0,:], args=(ka, k10, k12, k21), t_eval=t)

solved_gi = scipy_solved.y[0]
solved_c = scipy_solved.y[1]
solved_p = scipy_solved.y[2]

plt.figure(figsize = (12, 8))

plt.subplot(121)
plt.plot(t, gi, 'b--', label="GI - Euler's")
plt.plot(t, c, 'r--', label="Central - Euler's")
plt.plot(t, p, 'g--', label="Peripheral - Euler's")
plt.title("Amount of Drug Over Time - Euler's")
plt.xlabel('Time')
plt.ylabel('Amount of Drug (mg)')
plt.legend()
plt.grid()

plt.subplot(122)
plt.plot(t, solved_gi, 'b', label='GI - SciPy Integrate')
plt.plot(t, solved_c, 'r', label='Central - SciPy Integrate')
plt.plot(t, solved_p, 'g', label='Peripheral - SciPy Integrate')
plt.title('Amount of Drug Over Time - SciPy Integrate')
plt.xlabel('Time')
plt.ylabel('Amount of Drug (mg)')
plt.legend()
plt.grid()

plt.show()