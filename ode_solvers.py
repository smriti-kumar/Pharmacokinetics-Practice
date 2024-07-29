def euler_func_one_step(initial_condition, model_func, time, time_step, inputs, parameters):
    return initial_condition + time_step * model_func(initial_condition, time, parameters, inputs)