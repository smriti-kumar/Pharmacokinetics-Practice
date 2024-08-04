import numpy as np
import matplotlib.pyplot as plt
from t2d_model import generate_parameters, generate_initial_conditions, generate_inputs, model
from ode_solvers import euler_func_one_step

""" Generate Parameters """

model_parameters = generate_parameters(n_subjects = 10)

""" Generate Initial Conditions """

initial_conditions = generate_initial_conditions(parameters = model_parameters)


""" Generate Inputs """

time_step = 1
time = np.arange(0, 24 * 60, time_step)

model_inputs = generate_inputs(time = time, time_step=time_step, meal_times = np.array([6*60, 12 * 60, 18 * 60]), meal_amounts=np.array([100, 100, 100]))


print(model_parameters.shape, initial_conditions.shape)

""" Simulate Model """

solutions = np.zeros((time.size, initial_conditions.shape[0], initial_conditions.shape[1]))
print(solutions.shape)
solutions[0] = initial_conditions

for idx in range(1, time.size):

    solutions[idx] = euler_func_one_step(initial_condition=solutions[idx - 1],
                                        model_func=model,
                                        time = time[idx],
                                        time_step = time_step,
                                        inputs = model_inputs[idx],
                                        parameters=model_parameters)
    
# Convert from mmol/L to mg/dL
#solutions.T[4] *= 18
    

plt.figure()

mean = solutions.T[4].mean(axis = 0)
std = solutions.T[4].std(axis = 0)

plt.plot(time, mean)
plt.fill_between(time, mean - std, mean + std, alpha = 0.5)

plt.xlabel('Time (min)')
plt.ylabel('BGC (mmol/L)')

plt.show()
