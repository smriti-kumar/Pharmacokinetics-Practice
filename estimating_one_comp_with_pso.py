import numpy as np
from abstract_base_class_models import OneCompartmentModel
from helper_methods import euler_func
from tqdm import tqdm

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
    return [inputs[i] - 1 * ka * gi,
            ka * gi - k * b]

for i in range(0, len(t) - 1):
    gi, b = gb[i]
    agi, ab = adrug(t[i], gb[i], ka, k, inputs, i)
    gb[i + 1] = [gi + h * agi, b + h * ab]

noise_level = 0.1
gb_noisy = gb + noise_level * np.random.normal(size=gb.shape)

n = 30
location = np.random.rand(2, n) * 5

def objective_function(ka, k):
    global n, h, t, gb_noisy, inputs
    parameters = np.column_stack((ka, k)).T
    ka = np.clip(ka, 0, 1e2)
    k = np.clip(k, 0, 1e2)
    if (parameters.shape[1] == n):
        gb_sim = np.zeros((len(t), 2, n))
        error = np.zeros(n)
        onecomp_sim = OneCompartmentModel(gb_sim, parameters, inputs, 2)
        for i in range(0, len(t) - 1):
            gb_sim[i+1] = euler_func(gb_sim[i], onecomp_sim.ab(i).T, h)
        for i in range(n):
            gb_sim[:, :, i] = np.nan_to_num(gb_sim[:, :, i])
            error[i] = np.sum((gb_noisy - gb_sim[:, :, i])**2)
    else:
        gb_sim = np.zeros((len(t), 2))
        for i in range(0, len(t) - 1):
            gi, b = gb_sim[i]
            agi, ab = adrug(t[i], gb_sim[i], ka, k, inputs, i)
            gb_sim[i + 1] = [gi + h * agi, b + h * ab]
            gb_sim = np.nan_to_num(gb_sim)
        error = np.sum((gb_noisy - gb_sim)**2)
    return error

func_values = objective_function(location[0], location[1])
p_best = func_values
p_best_values = location
g_best = np.min(func_values)
g_best_values = p_best_values.argmin()

w = 0.8
c1 = 0.25
c2 = 0.25

V = np.random.randn(2, n) * 0.1

loop_n = int(1e6)
for i in tqdm(range(loop_n), desc = '>>> Running PSO...'):
    r0 = np.random.rand(1, n)
    r1 = np.random.rand(1, n)
    V = w * V + c1 * r0 * (p_best - location) + c2 * r1 * (g_best - location)
    location = location + V
    func_values = objective_function(location[0], location[1])
    p_best_values[:, (p_best >= func_values)] = location[:, (p_best >= func_values)]
    p_best = objective_function(p_best_values[0], p_best_values[1])
    g_best_values = p_best_values[:, p_best.argmin()]
    g_best = objective_function(g_best_values[0], g_best_values[1])

ka_sim, k_sim = g_best_values

print("ka =", ka_sim)
print("k =", k_sim)
