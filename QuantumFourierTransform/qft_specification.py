import math, cmath
import numpy as np

def program_specification_angles(qubitIndex, n, number, if_swap):
    '''
        return theta for Rz gate
        note that before applying Rz gate, Ry gate is required
    '''
    if if_swap:
        index = n - 1 - qubitIndex
    else:
        index = qubitIndex
    
    beta = 2 * math.pi * number / (2 ** (index + 1))
    return beta

def program_specification_state(n, number, if_swap):
    '''
        return the state vector
    '''
    for index in range(n):
        theta = 2 * math.pi * number / (2 ** (index + 1))
        temp_vec = 1 / math.sqrt(2) * np.array([1, cmath.exp(theta * 1j)])
        if index == 0:
            final_state = temp_vec
        else:
            if if_swap:
                final_state = np.kron(final_state, temp_vec)
            else:
                final_state = np.kron(temp_vec, final_state)
    return final_state
