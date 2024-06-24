import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import solve_ivp

h = 0.01 #step size
t = np.arange(0, 5 + h, h) #time values

#amount of drug in gi and body
def adrug(t, gb, ka, k):
    gi, b = gb
    return [- 1 * ka * gi,
            ka * gi - k * b]

d = 100 #mg
ka = 3 #absorption constant
gi0 = d #initial condition
b0 = 0 #initial condition
k = 2 #elimination constant

gb = np.zeros((len(t), 2))
gb[0,:] = [gi0, b0]

for i in range(0, len(t) - 1):
    gi, b = gb[i]
    agi, ab = adrug(t[i], gb[i], ka, k)
    gb[i + 1] = [gi + h * agi, b+ h * ab]

gi = gb[:, 0]
b = gb[:, 1]

scipy_solved = solve_ivp(adrug, (0, 5), gb[0,:], args=(ka, k), t_eval=t)

solved_gi = scipy_solved.y[0]
solved_b = scipy_solved.y[1]

plt.figure(figsize = (12, 8))

plt.subplot(121)
plt.plot(t, gi, 'b--', label="GI - Euler's")
plt.plot(t, b, 'r--', label="Body - Euler's")
plt.title("Amount of Drug Over Time - Euler's")
plt.xlabel('Time')
plt.ylabel('Amount of Drug (mg)')
plt.legend()
plt.grid()

plt.subplot(122)
plt.plot(t, solved_gi, 'b', label='GI - SciPy Integrate')
plt.plot(t, solved_b, 'r', label='Body - SciPy Integrate')
plt.title('Amount of Drug Over Time - SciPy Integrate')
plt.xlabel('Time')
plt.ylabel('Amount of Drug (mg)')
plt.legend()
plt.grid()

plt.show()
