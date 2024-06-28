def euler_func(initial_conditions, eq_function, time_step):
    """
    performs one step of euler's method.
    inputs:
        initial_conditions of function
        eq_function representing the diff eq
        time_step for euler step
    outputs:
        update after one time step
    """
    return initial_conditions + time_step * eq_function