import numpy as np
import matplotlib.pyplot as plt

h = 0.01 #step size
t = np.arange(0, 5 + h, h) #time values

#amount of drug in gastrointestinal tract
def agi(t, gi):
    return - 1 * ka * gi

#amount of drug in body
def ab (t, gi, b):
    return ka * gi - k * b

d = 100 #mg
ka = 3 #absorption constant
gi0 = d #initial condition
b0 = 0 #initial condition
k = 2 #elimination constant

gi = np.zeros(len(t))
gi[0] = gi0
b = np.zeros(len(t))
b[0] = b0

for i in range(0, len(t) - 1):
    gi[i + 1] = gi[i] + h * agi(t[i], gi[i])
    b[i + 1] = b[i] + h * ab(t[i], gi[i], b[i])

plt.figure(figsize = (12, 8))
plt.plot(t, b, 'r--', label='Approximate')
plt.plot(t, (d * ka * (np.exp(-1 * k * t) - np.exp(-1 * ka * t))) / (ka - k), 'b', label='Exact')
plt.title('Amount of Drug in Body Over Time')
plt.xlabel('Time')
plt.ylabel('Amount of Drug in Body')
plt.legend()
plt.grid()
plt.show()
