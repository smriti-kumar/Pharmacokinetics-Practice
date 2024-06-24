import numpy as np
import matplotlib.pyplot as plt
<<<<<<< HEAD
import scipy
from scipy.integrate import solve_ivp
=======
>>>>>>> ac8290694fc74329391800425ac0d5ed74b6d2dd

h = 0.01 #step size
t = np.arange(0, 5 + h, h) #time values

<<<<<<< HEAD
#amount of drug in gi and three compartments
def ab(t, gcpd, ka, k10, k12, k21, k13, k31):
    gi, c, p, dt = gcpd
    return [- 1 * ka * gi,
            ka * gi + k21 * p + k31 * dt - k10 * c - k12 * c - k13 * dt,
            k12 * c - k21 * p,
            k13 * c - k31 * dt]  
=======
#amount of drug in gastrointestinal tract
def agi(t, gi):
    return - 1 * ka * gi

#amount of drug in central compartment
def ac (t, gi, c, p, dt):
    return ka * gi + k21 * p + k31 * dt - k10 * c - k12 * c - k13 * dt

#amount of drug in peripheral compartment
def ap (t, p, c):
    return k12 * c - k21 * p

#amount of drug in deep tissue compartment
def adt (t, dt, c):
    return k13 * c - k31 * dt
>>>>>>> ac8290694fc74329391800425ac0d5ed74b6d2dd

d = 100 #mg
gi0 = d #initial condition for gi
c0 = 0 #initial condition for central
p0 = 0 #initial condition for peripheral
dt0 = 0 #initial condition for deep tissue
ka = 4 #absorption constant
k10 = 3 #elimination constant
k12 = 2 #central to peripheral constant
k21 = 1 #peripheral to central constant
<<<<<<< HEAD
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
                
=======
k13 = 2 #central to deep tissue constant
k31 = 1 #deep tissue to central constant

gi = np.zeros(len(t))
gi[0] = gi0
c = np.zeros(len(t))
c[0] = c0
p = np.zeros(len(t))
p[0] = p0
dt = np.zeros(len(t))
dt[0] = dt0

for i in range(0, len(t) - 1):
    gi[i + 1] = gi[i] + h * agi(t[i], gi[i])
    c[i + 1] = c[i] + h * ac(t[i], gi[i], c[i], p[i], dt[i])
    p[i + 1] = p[i] + h * ap(t[i], p[i], c[i])
    dt[i + 1] = dt[i] + h * adt(t[i], dt[i], c[i])

plt.figure(figsize = (12, 8))
plt.plot(t, c, 'r--', label='Approximate')
plt.plot(t, (d * ka * (np.exp(-1 * k10 * t) - np.exp(-1 * ka * t))) / (ka - k10), 'b', label='Exact')
plt.title('Amount of Drug in Central Compartment Over Time')
plt.xlabel('Time')
plt.ylabel('Amount of Drug in Body')
plt.legend()
plt.grid()
>>>>>>> ac8290694fc74329391800425ac0d5ed74b6d2dd
plt.show()
