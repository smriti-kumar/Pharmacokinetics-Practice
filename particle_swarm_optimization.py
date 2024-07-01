import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, ImageMagickWriter

def function(x, y):
    return np.sin(x**2) + np.cos(y**2) + 0.5 * x ** 2 + y ** 2

n = 20
location = np.random.rand(2, n) * 4 - 2

vectorized_function = np.vectorize(function)
func_values = vectorized_function(location[0], location[1])
p_best = func_values
p_best_values = location
g_best = np.min(func_values)
g_best_values = p_best_values.argmin()

w = 0.8
c1 = 0.25
c2 = 0.25

V = np.random.randn(2, n) * 0.1

loop_n = 30
for i in range (loop_n):
    r0 = np.random.rand()
    r1 = np.random.rand()
    V = w * V + c1 * r0 * (p_best - location) + c2 * r1 * (g_best.reshape(-1,1) - location)
    location = location + V
    func_values = vectorized_function(location[0], location[1])
    p_best_values[:, (p_best >= func_values)] = location[:, (p_best >= func_values)]
    p_best = vectorized_function(p_best_values[0], p_best_values[1])
    g_best_values = p_best_values[:, p_best.argmin()]
    g_best = vectorized_function(g_best_values[0], g_best_values[1])

print(g_best_values)
print(g_best)
