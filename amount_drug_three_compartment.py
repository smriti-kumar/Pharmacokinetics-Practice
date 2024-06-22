import numpy as np
import matplotlib.pyplot as plt

h = 0.01 #step size
t = np.arange(0, 5 + h, h) #time values

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

d = 100 #mg
gi0 = d #initial condition for gi
c0 = 0 #initial condition for central
p0 = 0 #initial condition for peripheral
dt0 = 0 #initial condition for deep tissue
ka = 4 #absorption constant
k10 = 3 #elimination constant
k12 = 2 #central to peripheral constant
k21 = 1 #peripheral to central constant
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
plt.show()
