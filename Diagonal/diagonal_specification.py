import math, cmath
import numpy as np

def program_specification_angles(qubitIndex, initialState, diagList):
    '''
        RT return theta for Ry gate, beta for Rz gate
    '''
    selInitStatStr = initialState[qubitIndex]
    if selInitStatStr == '|0>':
        selInitStat = [1, 0]
    elif selInitStatStr == '|1>':
        selInitStat = [0, 1]
    elif selInitStatStr == '|+>':
        selInitStat = [1 / math.sqrt(2), 1 / math.sqrt(2)]
    elif selInitStatStr == '|->':
        selInitStat = [1 / math.sqrt(2), -1 / math.sqrt(2)]
    diagOperation = diagList[qubitIndex]
    selFinaStat = [selInitStat[0] * diagOperation[0],
                   selInitStat[1] * diagOperation[1]]
    beta = cmath.phase(selFinaStat[1]) - cmath.phase(selFinaStat[0])
    theta = 2 * math.acos(abs(selFinaStat[0]))
    
    return beta, theta

def program_specification_state(initialState, diagList):
    for index in range(len(initialState)):
        selInitStatStr = initialState[-1-index]
        diagOperation = diagList[-1-index]
        
        if selInitStatStr == '|0>':
            selInitStat = [1, 0]
        elif selInitStatStr == '|1>':
            selInitStat = [0, 1]
        elif selInitStatStr == '|+>':
            selInitStat = [1 / math.sqrt(2), 1 / math.sqrt(2)]
        elif selInitStatStr == '|->':
            selInitStat = [1 / math.sqrt(2), -1 / math.sqrt(2)]
        
        tempFinaStat = [selInitStat[0] * diagOperation[0],
                        selInitStat[1] * diagOperation[1]]
        if index == 0:
            finalStat = np.array(tempFinaStat)
        else:
            finalStat = np.kron(tempFinaStat, finalStat)
    return finalStat
