from qiskit import QuantumCircuit
import numpy as np
import time
from tqdm import tqdm

import ast
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from statistical_tests import *
from preprossing import *
from circuit_execution import *

from diagonal_specification import *
from diagonal_defect1 import Diagonal_defect1
from diagonal_defect2 import Diagonal_defect2
from diagonal_defect3 import Diagonal_defect3
from diagonal_defect4 import Diagonal_defect4
from diagonal_defect5 import Diagonal_defect5

program_name = 'Diagonal'

def version_selection(program_name, program_version):
    '''
        select the quantum program to be tested

        Input variable:
            + program_name       [str] e.g. "IntegerComparator"
            + program_version    [str] including "v1", "v2", "v3", "v4", "v5"
    '''
    if program_version[0] == "v":
        function_name = program_name + '_defect' + program_version[1:]
    else:
        return f"Invalid program program_."

    if function_name in globals():
        func = globals()[function_name]
        return func
    else:
        return f"Function '{function_name}' not found."

def testing_with_statistical_methods(df, 
                                     program_version, 
                                     total_repeats, 
                                     shot_list, 
                                     statistical_method,
                                     toler_err=0.05):
    '''
        This function implements the test process using statistical methods
        
        Input variables:
            + df:                            the dataset for test suites
            + program_version:      [str]    the buggy version to be tested, including "v1", "v2", "v3", "v4" and "v5"
            + total_repeats:        [int]    the parameter $r$ involved in the paper standing for independent repetitive  
                                             executions of the same test process  
            + shot_list:            [list]   the list of configured shots $s$ 
            + statistical_methods:  [str]    the concrete OPO, including "MWTest", "ChiTest", "KSTest", "CrsEnt" and 
                                             "JSDiv"
            + toler_err:            [float]  the tolerable error limit for OPOs, such as p-value for NHTs and distance
                                             threshold for SDMs

        Output variables:
            + time_records:         [list]   the execution time for $r$ repeats
            + fault_records:        [list]   the detected faults for $r$ repeats
    '''
    time_records = []                    # record the average time
    fault_records = []                   # record the average faults
    print(statistical_method)
    for shots in tqdm(shot_list):                    
        time_list = []                            
        failure_list = []
        for index in range(total_repeats):           # repeat for statistically significant results
            start_time = time.time()                 # record time                
            failures = 0 
            for test_order in range(len(df)):
                test_input = df.iloc[test_order] 
                # need to vary with different programs
                num_qubits = test_input.iloc[1]
                initial_state, realDiagPair, imagDiagPair = test_input.iloc[2], test_input.iloc[3], test_input.iloc[4]
                initial_state = ast.literal_eval(initial_state)
                realDiagPair = ast.literal_eval(realDiagPair)
                imagDiagPair= ast.literal_eval(imagDiagPair)
                
                diag_pair_list = realDiagPair.copy()
                for index in range(len(diag_pair_list)):
                    diag_pair_list[index] = [complex(realDiagPair[index][0], imagDiagPair[index][0]),
                                            complex(realDiagPair[index][1], imagDiagPair[index][1])]
                
                # calculate kron product
                diag_matrix_array = [np.array(elem) for elem in diag_pair_list]
                temp_matrix = diag_matrix_array[0]
                for diag in diag_matrix_array[1:]:
                    temp_matrix = np.kron(temp_matrix, diag)
                diag_matrix = list(temp_matrix)

                # state preparation
                qc_initial = QuantumCircuit(num_qubits, num_qubits)
                for index, val in enumerate(initial_state[::-1]):
                    if val == '|1>':
                        qc_initial.x(index)
                    elif val == '|+>':
                        qc_initial.h(index)
                    elif val == '|->':
                        qc_initial.x(index)
                        qc_initial.h(index)
                            
                # running the quantum programs
                qc = qc_initial.copy()
                func = version_selection(program_name, program_version)
                qc_test = func(diag_matrix)
                qc.append(qc_test, qc.qubits)
                qc.measure(qc.qubits,qc.clbits)

                dict_counts = circuit_execution(qc, shots)
                
                # transform a dict into a list
                test_samples = []
                for (key, value) in dict_counts.items():
                    test_samples += [key] * value
                
                # generate expected sample according to the expected probabilities
                exp_state = program_specification_state(initial_state, diag_pair_list)
                exp_probs = list(abs(np.array(exp_state)) ** 2) 
               
                if statistical_method not in ['CrsEnt', 'JSDiv']:     # NHTs: generate samples comforting the expected probabilities
                    exp_outputs = list(np.random.choice(range(2 ** num_qubits), size=shots, p=exp_probs))
                else:                                       # OPOs: use the probability distribution
                    exp_outputs = exp_probs
                test_result = statistical_method_selection(num_qubits, 
                                                           exp_outputs, 
                                                           test_samples, 
                                                           statistical_method, 
                                                           toler_err=toler_err)

                failures += int(test_result == 'fail')                   
                        
            durTime = time.time() - start_time            
            time_list += [durTime]
            failure_list += [failures]                  
        time_records.append(time_list)
        fault_records.append(failure_list)       
    return time_records, fault_records

def testing_with_STFQ(df, program_version, total_repeats, shot_list):
    '''
        This function implements the test process using STFQ (Swap Test on Full Qubits)
        
        Input variables:
            + df:                            the dataset for test suites
            + program_version:      [str]    the buggy version to be tested, including "v1", "v2", "v3", "v4" and "v5"
            + total_repeats:        [int]    the parameter $r$ involved in the paper standing for independent repetitive  
                                             executions of the same test process  
            + shot_list:            [list]   the list of configured shots $s$ 

        Output variables:
            + time_records:         [list]   the execution time for $r$ repeats
            + fault_records:        [list]   the detected faults for $r$ repeats
    '''
    time_records = []        
    fault_records = []         

    print("STFQ")
    for shots in tqdm(shot_list):                    # variable shots
        time_list = []                               # record time
        failure_list = []
        for index in range(total_repeats):          # repeat for statistically significant results
            start_time = time.time()                 # record time                
            failures = 0 
            for test_order in range(len(df)): 
                test_input = df.iloc[test_order]                   
                
                # need to vary with different programs
                num_qubits = test_input.iloc[1]
                initial_state, realDiagPair, imagDiagPair = test_input.iloc[2], test_input.iloc[3], test_input.iloc[4]
                initial_state = ast.literal_eval(initial_state)
                realDiagPair = ast.literal_eval(realDiagPair)
                imagDiagPair= ast.literal_eval(imagDiagPair)
                
                diag_pair_list = realDiagPair.copy()
                for index in range(len(diag_pair_list)):
                    diag_pair_list[index] = [complex(realDiagPair[index][0], imagDiagPair[index][0]),
                                            complex(realDiagPair[index][1], imagDiagPair[index][1])]
                
                qc_initial = QuantumCircuit(2 * num_qubits + 1, 1)

                # calculate kron product
                diag_matrix_array = [np.array(elem) for elem in diag_pair_list]
                for index, diag in enumerate(diag_matrix_array[::-1]):
                    if index == 0:
                        temp_matrix = np.array(diag)
                    else:
                        temp_matrix = np.kron(diag, temp_matrix)
                diag_matrix = list(temp_matrix)

                # state preparation
                for index, val in enumerate(initial_state[::-1]):
                    if val == '|1>':
                        qc_initial.x(index)
                    elif val == '|+>':
                        qc_initial.h(index)
                    elif val == '|->':
                        qc_initial.x(index)
                        qc_initial.h(index)

                qc = qc_initial.copy()
                # append the tested subroutine
                func = version_selection(program_name, program_version)
                qc_test = func(diag_matrix)
                # prepare the expected state
                qc_exp = QuantumCircuit(num_qubits)
                exp_state = program_specification_state(initial_state, diag_pair_list)
                qc_exp.initialize(exp_state, qc_exp.qubits)
                qc.append(qc_test, qc.qubits[:num_qubits])
                qc.append(qc_exp, qc.qubits[num_qubits:2*num_qubits])
                qc.h(-1)
                for i in range(num_qubits):
                    qc.cswap(-1, i, i + num_qubits)
                qc.h(-1)
                qc.measure(qc.qubits[-1],qc.clbits)
                    
                dict_counts = circuit_execution(qc, shots)
                
                # transform a dict into a list
                resList = list(dict_counts.keys())
                if len(resList) == 1 and resList[0] == 0:
                    test_result = 'pass'
                else:
                    test_result = 'fail'
                failures += int(test_result == 'fail')                   
                        
            durTime = time.time() - start_time            
            time_list += [durTime]
            failure_list += [failures]               
        time_records.append(time_list)
        fault_records.append(failure_list)     
    return time_records, fault_records

def testing_with_STSQ(df, program_version, total_repeats, shot_list):    
    '''
        This function implements the test process using STSQ (Swap Test on Separate Qubits)
        
        Input variables:
            + df:                            the dataset for test suites
            + program_version:      [str]    the buggy version to be tested, including "v1", "v2", "v3", "v4" and "v5"
            + total_repeats:        [int]    the parameter $r$ involved in the paper standing for independent repetitive  
                                             executions of the same test process  
            + shot_list:            [list]   the list of configured shots $s$ 

        Output variables:
            + time_records:         [list]   the execution time for $r$ repeats
            + fault_records:        [list]   the detected faults for $r$ repeats
    '''
    time_records = []          
    fault_records = []       

    print('STSQ')
    for shots in tqdm(shot_list):           # variance of shots
        time_list = []                # record time
        failure_list = []
        for index in range(total_repeats):     # repeat for statistically significant results
            start_time = time.time()
            failures = 0                 
            for test_order in range(len(df)):
                test_input = df.iloc[test_order]                   
                
                # need to vary with different programs
                num_qubits = test_input.iloc[1]
                initial_state, realDiagPair, imagDiagPair = test_input.iloc[2], test_input.iloc[3], test_input.iloc[4]
                initial_state = ast.literal_eval(initial_state)
                realDiagPair = ast.literal_eval(realDiagPair)
                imagDiagPair= ast.literal_eval(imagDiagPair)
                
                diag_pair_list = realDiagPair.copy()
                for index in range(len(diag_pair_list)):
                    diag_pair_list[index] = [complex(realDiagPair[index][0], imagDiagPair[index][0]),
                                            complex(realDiagPair[index][1], imagDiagPair[index][1])]
                
                qc_initial = QuantumCircuit(num_qubits + 2, 1)

                # calculate kron product
                diag_matrix_array = [np.array(elem) for elem in diag_pair_list]
                for index, diag in enumerate(diag_matrix_array[::-1]):
                    if index == 0:
                        temp_matrix = np.array(diag)
                    else:
                        temp_matrix = np.kron(diag, temp_matrix)
                diag_matrix = list(temp_matrix)

                # state preparation
                for index, val in enumerate(initial_state[::-1]):
                    if val == '|1>':
                        qc_initial.x(index)
                    elif val == '|+>':
                        qc_initial.h(index)
                    elif val == '|->':
                        qc_initial.x(index)
                        qc_initial.h(index)
                
                # select a qubit to test 
                qubitTestList = np.random.choice(range(num_qubits), 
                                                size=num_qubits, 
                                                replace=False)
                
                test_result = 'pass'
                func = version_selection(program_name, program_version)
                qc_test = func(diag_matrix)
                qc_initial.append(qc_test, qc_initial.qubits[:num_qubits])                    
                for selOrder, tempQubit in enumerate(qubitTestList):                          
                    qc = qc_initial.copy()
                    beta, theta = program_specification_angles(tempQubit, initial_state, diag_pair_list)
                    qc.ry(theta, -2)
                    qc.rz(beta, -2)
                    qc.h(-1)
                    qc.cswap(-1, -3-tempQubit, -2)
                    qc.h(-1)
                    qc.measure(-1, qc.clbits)

                    dict_counts = circuit_execution(qc, shots)
            
                    # transform a dict into a list
                    resList = list(dict_counts.keys())
                    
                    # step += 1
                    if len(resList) == 1 and resList[0] == 0:   # this qubit passes                     
                        continue
                            
                    else:                                       # this qubit fails
                        test_result = 'fail'
                        failures += 1 
                        break

            durTime = time.time() - start_time          
            time_list += [durTime]
            failure_list += [failures]
        time_records.append(time_list)
        fault_records.append(failure_list)     
    
    return time_records, fault_records

def testing_with_HOSS(df, program_version, total_repeats, shot_list):
    '''
        This function implements the test process using HOSS (Hybrid Oracle via Separable States)
        
        Input variables:
            + df:                            the dataset for test suites
            + program_version:      [str]    the buggy version to be tested, including "v1", "v2", "v3", "v4" and "v5"
            + total_repeats:        [int]    the parameter $r$ involved in the paper standing for independent repetitive  
                                             executions of the same test process  
            + shot_list:            [list]   the list of configured shots $s$ 

        Output variables:
            + time_records:         [list]   the execution time for $r$ repeats
            + fault_records:        [list]   the detected faults for $r$ repeats
    '''
    time_records = []         
    fault_records = []      
    
    print("HOSS")
 
    for shots in tqdm(shot_list):               # variance of shots
        time_list = []                          # record time
        failure_list = []
        for index in range(total_repeats):      # repeat for statistically significant results
            start_time = time.time()
            failures = 0
            for test_order in range(len(df)): 
                test_input = df.iloc[test_order]                   
                
                # need to vary with different programs
                num_qubits = test_input.iloc[1]
                initial_state, realDiagPair, imagDiagPair = test_input.iloc[2], test_input.iloc[3], test_input.iloc[4]
                initial_state = ast.literal_eval(initial_state)
                realDiagPair = ast.literal_eval(realDiagPair)
                imagDiagPair= ast.literal_eval(imagDiagPair)
                
                diag_pair_list = realDiagPair.copy()
                for index in range(len(diag_pair_list)):
                    diag_pair_list[index] = [complex(realDiagPair[index][0], imagDiagPair[index][0]),
                                             complex(realDiagPair[index][1], imagDiagPair[index][1])]
                
                # calculate kron product
                diag_matrix_array = [np.array(elem) for elem in diag_pair_list]
                for index, diag in enumerate(diag_matrix_array[::-1]):
                    if index == 0:
                        temp_matrix = np.array(diag)
                    else:
                        temp_matrix = np.kron(diag, temp_matrix)
                diag_matrix = list(temp_matrix)
                
                # cluster 
                qubitPartition = [0] * int(num_qubits)   # the partition index corresponding to each qubit
                numPartition = 0
                for index, val in enumerate(initial_state):
                    if val == '|+>' or val == '|->':
                        numPartition += 1
                        qubitPartition[index] = numPartition

                set_partitions = list(set(qubitPartition))
                clusterOrder = np.random.choice(set_partitions, 
                                                size=len(set_partitions), 
                                                replace=False)
                test_result = 'pass'
                for cluster in clusterOrder:
                    # cluster == 0 refers to nonsuperposition 
                    if cluster == 0:
                        num_classical_state = qubitPartition.count(0)
                        qc = QuantumCircuit(num_qubits, num_classical_state)

                        # state preparation (from high to low)
                        for index, val in enumerate(initial_state[::-1]):
                            if val == '|1>':
                                qc.x(index)
                            elif val == '|+>':
                                qc.h(index)
                            elif val == '|->':
                                qc.x(index)
                                qc.h(index)
                            
                        # running the quantum programs
                        # need to vary with different programs
                        func = version_selection(program_name, program_version)
                        qc_test = func(diag_matrix)
                        qc.append(qc_test, qc.qubits)

                        # measurement
                        clbitsIndex = 0
                        expBinaList = []
                        for index, val in enumerate(qubitPartition):
                            if val == 0:
                                qc.measure(qc.qubits[-1-index], qc.clbits[-1-clbitsIndex])
                                clbitsIndex += 1
                                if initial_state[index] == '|0>':
                                    expBinaList = expBinaList + [0]
                                elif initial_state[index] == '|1>':
                                    expBinaList = expBinaList + [1]
                        
                        # convert to binary number 
                        binaStr = ''.join(map(str, expBinaList))
                        number = int(binaStr, 2)
                        dict_counts = circuit_execution(qc, shots)
                        
                        # transform a dict into a list
                        resList = list(dict_counts.keys())
                        if len(resList) == 1 and resList[0] == number:
                            continue        # test_result = 'pass'
                        else:
                            test_result = 'fail'
                            break
    
                    elif cluster != 0:
                        qc = QuantumCircuit(num_qubits + 2, 1)

                        # state preparation
                        for index, val in enumerate(initial_state[::-1]):
                            if val == '|1>':
                                qc.x(index)
                            elif val == '|+>':
                                qc.h(index)
                            elif val == '|->':
                                qc.x(index)
                                qc.h(index)

                        # running the quantum programs
                        # need to vary with different programs
                        func = version_selection(program_name, program_version)
                        qc_test = func(diag_matrix)

                        qc.append(qc_test, qc.qubits[:num_qubits])
                        selQubitInd = qubitPartition.index(cluster)
                        beta, theta = program_specification_angles(selQubitInd, initial_state, diag_pair_list)
                        qc.ry(theta, -2)
                        qc.rz(beta, -2)
                        qc.h(-1)
                        qc.cswap(-1, -3-selQubitInd, -2)
                        qc.h(-1)
                        qc.measure(-1, qc.clbits)
                        dict_counts = circuit_execution(qc, shots)
                        
                        # transform a dict into a list
                        resList = list(dict_counts.keys())
                        
                        if len(resList) == 1 and resList[0] == 0:   # this qubit passes                     
                            continue        # test_result = 'pass'            
                        else:                                       # this qubit fails
                            test_result = 'fail'
                            break

                failures += int(test_result == 'fail')                   
            durTime = time.time() - start_time            
            time_list += [durTime]
            failure_list += [failures]
        time_records.append(time_list)
        fault_records.append(failure_list)   
    return time_records, fault_records