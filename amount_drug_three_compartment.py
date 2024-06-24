import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import solve_ivp

h = 0.01 #step size
t = np.arange(0, 5 + h, h) #time values

#amount of drug in gi and three compartments
def ab(t, gcpd, ka, k10, k12, k21, k13, k31):
    gi, c, p, dt = gcpd
    return [- 1 * ka * gi,
            ka * gi + k21 * p + k31 * dt - k10 * c - k12 * c - k13 * dt,
            k12 * c - k21 * p,
            k13 * c - k31 * dt]  

d = 100 #mg
gi0 = d #initial condition for gi
c0 = 0 #initial condition for central
p0 = 0 #initial condition for peripheral
dt0 = 0 #initial condition for deep tissue
ka = 4 #absorption constant
k10 = 3 #elimination constant
k12 = 2 #central to peripheral constant
k21 = 1 #peripheral to central constant
k13 = 0.8 #central to deep tissue constant
k31 = 0.4 #deep tissue to central constant

gcpd = np.zeros((len(t), 4))
gcpd[0,:] = [gi0, c0, p0, dt0]

for i in range(0, len(t) - 1):
    gi, c, p, dt = gcpd[i]
    agi, ac, ap, adt = ab(t[i], gcpd[i], ka, k10, k12, k21, k13, k31)
    gcpd[i + 1] = [gi + h * agi, c + h * ac, p + h * ap, dt + h * adt]

gi = gcpd[:, 0]
c = gcpd[:, 1]
p = gcpd[:, 2]
dt = gcpd[:, 3]

scipy_solved = solve_ivp(ab, (0, 5), gcpd[0,:], args=(ka, k10, k12, k21, k13, k31), t_eval=t)

solved_gi = scipy_solved.y[0]
solved_c = scipy_solved.y[1]
solved_p = scipy_solved.y[2]
solved_dt = scipy_solved.y[3]

plt.subplot(121)
plt.plot(t, gi, 'b--', label="GI - Euler's")
plt.plot(t, c, 'r--', label="Central - Euler's")
plt.plot(t, p, 'g--', label="Peripheral - Euler's")
plt.plot(t, dt, 'm--', label="Deep Tissue - Euler's")
plt.title("Amount of Drug Over Time - Euler's")
plt.xlabel('Time')
plt.ylabel('Amount of Drug (mg)')
plt.legend()
plt.grid()


plt.subplot(122)
plt.plot(t, solved_gi, 'b', label='GI - SciPy Integrate')
plt.plot(t, solved_c, 'r', label='Central - SciPy Integrate')
plt.plot(t, solved_p, 'g', label='Peripheral - SciPy Integrate')
plt.plot(t, solved_dt, 'm', label='Deep Tissue - SciPy Integrate')
plt.title('Amount of Drug Over Time - SciPy Integrate')
plt.xlabel('Time')
plt.ylabel('Amount of Drug (mg)')
plt.legend()
plt.grid()
                
plt.show()