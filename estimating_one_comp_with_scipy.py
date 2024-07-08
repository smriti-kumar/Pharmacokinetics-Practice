import numpy as np
from abstract_base_class_models import OneCompartmentModel
from helper_methods import euler_func
import scipy

h = 0.01 #step size
total_time = 5 #total time
t = np.arange(0, total_time + h, h) #time values

ka = 3 #absorption constant
k = 2 #elimination constant

gb = np.zeros((len(t), 2))
inputs = np.zeros_like(t)
dosing_interval = np.argwhere(t % (100 * h) == 0)
for i in range(len(dosing_interval)):
    dosing_interval[i] += int(np.random.rand() * 30 - 15)
    if dosing_interval[i] > 500:
        dosing_interval[i] = 500
    elif dosing_interval[i] < 0:
        dosing_interval[i] = 0
inputs[dosing_interval] = 100 / h
parameters = np.array([ka, k])

def adrug(t, gb, ka, k, inputs, i):
    gi, b = gb.T
    return [inputs[i] - ka * gi,
            ka * gi - k * b]

for i in range(0, len(t) - 1):
    gi, b = gb[i]
    agi, ab = adrug(t[i], gb[i], ka, k, inputs, i)
    gb[i + 1] = [gi + h * agi, b + h * ab]

noise_level = 0.1
gb_noisy = gb + noise_level * np.random.normal(size=gb.shape)

def func(t, ka, k):
    global inputs, h
    gb_new = np.zeros((len(t), 2))
    for i in range(0, len(t) - 1):
        gi_new, b_new = gb_new[i]
        agi_new, ab_new = adrug(t[i], gb_new[i], ka, k, inputs, i)
        gb_new[i + 1] = [gi_new + h * agi_new, b_new + h * ab_new]
    return gb_new.flatten()
parameters_new, parameters_covariance = scipy.optimize.curve_fit(func, t, gb_noisy.flatten())
ka_new, k_new = parameters_new
print("ka =", ka_new)
print("k =", k_new)