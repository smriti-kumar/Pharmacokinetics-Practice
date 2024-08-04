import numpy as np
from helper_methods import euler_func
import scipy
import pandas as pd
import matplotlib.pyplot as plt
from t2d_model import generate_parameters, generate_initial_conditions, generate_inputs, model
from ode_solvers import euler_func_one_step

""" Model Fitting """

time_step = 0.35

df = pd.read_csv("Digitized Glucose Data for Liraglutide - placebo data.csv")
df = pd.DataFrame.to_numpy(df)
time = df[:, 0]
glucose_levels = df[:, 1]


def func(time, BW0, RTG0, GFR0, RTG_Max, EC50, ksa, ke, Vd, ka, Gprod0, Q, Vp, Vg, ClG, ClGI, kGE, GSS, ISS, IPRG, Sincr, Isec0, ClI, Vi, kIE):
    global time_step
    model_parameters = np.column_stack((BW0, \
                                        RTG0, \
                                        GFR0, \
                                        RTG_Max, \
                                        EC50, \
                                        ksa, \
                                        ke, \
                                        Vd, \
                                        ka, \
                                        Gprod0, \
                                        Q, \
                                        Vp, \
                                        Vg, \
                                        ClG, \
                                        ClGI, \
                                        kGE, \
                                        GSS, \
                                        ISS, \
                                        IPRG, \
                                        Sincr, \
                                        Isec0, \
                                        ClI, \
                                        Vi, \
                                        kIE))
    initial_conditions = generate_initial_conditions(parameters = model_parameters)
    model_inputs = generate_inputs(time = time, time_step=time_step, meal_times = np.array([0]), meal_amounts=np.array([0]))
    solutions = np.zeros((time.size, initial_conditions.shape[0], initial_conditions.shape[1]))
    solutions[0] = initial_conditions
    for idx in range(1, time.size):
        solutions[idx] = euler_func_one_step(initial_condition=solutions[idx - 1],
                                            model_func=model,
                                            time = time[idx],
                                            time_step = time_step,
                                            inputs = model_inputs[idx],
                                            parameters=model_parameters)
    return solutions[:, :, 4].flatten()

est_parameters, parameters_covariance = scipy.optimize.curve_fit(func, time, glucose_levels)
print("Parameters:", est_parameters)
BW0, RTG0, GFR0, RTG_Max, EC50, ksa, ke, Vd, ka, Gprod0, Q, Vp, Vg, ClG, ClGI, kGE, GSS, ISS, IPRG, Sincr, Isec0, ClI, Vi, kIE = est_parameters
est_parameters = np.column_stack((BW0, \
                                        RTG0, \
                                        GFR0, \
                                        RTG_Max, \
                                        EC50, \
                                        ksa, \
                                        ke, \
                                        Vd, \
                                        ka, \
                                        Gprod0, \
                                        Q, \
                                        Vp, \
                                        Vg, \
                                        ClG, \
                                        ClGI, \
                                        kGE, \
                                        GSS, \
                                        ISS, \
                                        IPRG, \
                                        Sincr, \
                                        Isec0, \
                                        ClI, \
                                        Vi, \
                                        kIE))
""" Estimation """

est_initial_conditions = generate_initial_conditions(parameters = est_parameters)
est_model_inputs = generate_inputs(time = time, time_step=time_step, meal_times = np.array([0]), meal_amounts=np.array([0]))
est_solutions = np.zeros((time.size, est_initial_conditions.shape[0], est_initial_conditions.shape[1]))
est_solutions[0] = est_initial_conditions
for idx in range(1, time.size):
    est_solutions[idx] = euler_func_one_step(initial_condition=est_solutions[idx - 1],
                                        model_func=model,
                                        time = time[idx],
                                        time_step = time_step,
                                        inputs = est_model_inputs[idx],
                                        parameters=est_parameters)


plt.plot(time, est_solutions[:, :, 4], 'b--', label='Estimated')
plt.plot(time, glucose_levels, 'r--', label='Given')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Glucose Levels (mmol/L)')
plt.title('Glucose Fitting')
plt.show()
