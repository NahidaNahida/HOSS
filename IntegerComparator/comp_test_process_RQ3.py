from qiskit import QuantumCircuit
import numpy as np
import time
from tqdm import tqdm

import ast
import sys, os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)

from statistical_tests import *
from preprossing import *
from circuit_execution import *

from comp_specification import *
from comp_defect1 import IntegerComparator_defect1
from comp_defect2 import IntegerComparator_defect2
from comp_defect3 import IntegerComparator_defect3
from comp_defect4 import IntegerComparator_defect4
from comp_defect5 import IntegerComparator_defect5

program_name = 'IntegerComparator'
file_short_name = 'IC'

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
    
def testing_with_statistical_methods(qubit_list, 
                                     program_version, 
                                     total_repeats, 
                                     shots, 
                                     statistical_method):
    '''
        This function implements the test process using statistical methods
        
        Input variables:
            + qubit_list:           [list]   the list of qubit numbers
            + program_version:      [str]    the buggy version to be tested, including "v1", "v2", "v3", "v4" and "v5"
            + total_repeats:        [int]    the parameter $r$ involved in the paper standing for independent repetitive  
                                             executions of the same test process  
            + shots:                [int]    the configured shots $s$ 
            + statistical_methods:  [str]    the concrete OPO, including "MWTest", "ChiTest", "KSTest", "CrsEnt" 
                                             and "JSDiv"

        Output variables:
            + time_records:         [list]   the execution time for $r$ repeats
            + fault_records:        [list]   the detected faults for $r$ repeats
            + err_records:          [list]   the estimated type I and type II errors
    '''
    time_records = []                    
    fault_records = []
    err_records = []      
    
    initial_gates = [0, 1]

    print(statistical_method)

    for num_qubits in tqdm(qubit_list):                    
        filename = '/' + file_short_name + '_' + str(program_version) + '_testSuites_(qubit=' \
            + str(num_qubits) + ',fr=0.5,#t=50).csv'
        df = pd.read_csv(parent_dir + filename, skiprows=0)
        err_record = np.array([0, 0])                   # err_record[0]: type I; err_record[1]: type II
        ora_record = np.array([0, 0])                   # ora_record[0]: "pass"; ora_record[1]: "fail"
        time_list = []
        failure_list = []
        for index in range(total_repeats):     # repeat for statistically significant results
            start_time = time.time()                 # record time
            failures = 0
            for test_order in range(len(df)): 
                test_input = df.iloc[test_order]                   
                
                # need to vary with different programs
                n, num_qubits = test_input.iloc[0], test_input.iloc[1]
                number, L, sign = test_input.iloc[2], test_input.iloc[3], test_input.iloc[4]

                exp_test_result = 'pass' if bool(test_input.iloc[-1]) else 'fail'
                if exp_test_result == 'pass':
                    ora_record[0] += 1
                elif exp_test_result == 'fail':
                    ora_record[1] += 1

                strNumber = bin(number)[2:]
                strNumber = strNumber.zfill(n)  
                initial_state = [int(bit) for bit in strNumber]
                
                qc_initial = QuantumCircuit(num_qubits, num_qubits)
                for index, val in enumerate(initial_state[::-1]):
                    if initial_gates[val] == 1:
                        qc_initial.x(index)
                            
                # running the quantum programs
                qc = qc_initial.copy()

                # need to vary with different programs
                func = version_selection(program_name, program_version)
                qc_test = func(n, L, sign)

                qc.append(qc_test, qc.qubits)
                qc.measure(qc.qubits,qc.clbits)
                    
                dict_counts = circuit_execution(qc, shots)
                
                # transform a dict into a list
                test_samples = []
                for (key, value) in dict_counts.items():
                    test_samples += [key] * value
                
                # generate expected sample according to the expected probabilities
                exp_value = program_specification_value(n, number, L, sign)
                exp_samples = [exp_value] * shots
                exp_probs = [0] * (2 ** num_qubits)
                exp_probs[exp_value] = 1

                if statistical_method not in ['CrsEnt', 'JSDiv']:     # NHTs: generate samples comforting the expected probabilities
                    exp_outputs = exp_samples
                else:                                       # OPOs: use the probability distribution
                    exp_outputs = exp_probs
                test_result = statistical_method_selection(num_qubits, 
                                                           exp_outputs, 
                                                           test_samples, 
                                                           statistical_method)
                failures += int(test_result == 'fail')                   
                        
                if exp_test_result == 'pass' and test_result == 'fail':
                    err_record[0] += 1
                elif exp_test_result == 'fail' and test_result == 'pass':
                    err_record[1] += 1    

            durTime = time.time() - start_time            
            time_list += [durTime]
            failure_list += [failures]                  
        time_records.append(time_list)
        fault_records.append(failure_list)
        err_record = err_record / ora_record
        err_records.append(list(err_record))

    return time_records, fault_records, err_records

def testing_with_STFQ(qubit_list, program_version, total_repeats, shots):
    '''
        This function implements the test process using STFQ (Swap Test on Full Qubits)
        
        Input variables:
            + qubit_list:           [list]   the list of qubit numbers
            + program_version:      [str]    the buggy version to be tested, including "v1", "v2", "v3", "v4" and "v5"
            + total_repeats:        [int]    the parameter $r$ involved in the paper standing for independent repetitive  
                                             executions of the same test process  
            + shots:                [int]    the configured shots $s$ 
            + statistical_methods:  [str]    the concrete OPO, including "MWTest", "ChiTest", "KSTest", "CrsEnt" 
                                             and "JSDiv"

        Output variables:
            + time_records:         [list]   the execution time for $r$ repeats
            + fault_records:        [list]   the detected faults for $r$ repeats
            + err_records:          [list]   the estimated type I and type II errors
    '''
    time_records = []         
    fault_records = []        
    err_records = []     
    initial_gates = [0, 1]

    print("STFQ")
 
    for num_qubits in tqdm(qubit_list):                   
        filename = '/' + file_short_name + '_' + str(program_version) + '_testSuites_(qubit=' \
            + str(num_qubits) + ',fr=0.5,#t=50).csv'
        df = pd.read_csv(parent_dir + filename, skiprows=0)
        err_record = np.array([0, 0])                   # err_record[0]: type I; err_record[1]: type II
        ora_record = np.array([0, 0])                   # ora_record[0]: "pass"; ora_record[1]: "fail"
        time_list = []
        failure_list = []
        for index in range(total_repeats):     # repeat for statistically significant results
            start_time = time.time()                 # record time                
            failures = 0 
            for test_order in range(len(df)): 
                test_input = df.iloc[test_order]                   
                
                # need to vary with different programs
                n, num_qubits = test_input.iloc[0], test_input.iloc[1]
                number, L, sign = test_input.iloc[2], test_input.iloc[3], test_input.iloc[4]

                exp_test_result = 'pass' if bool(test_input.iloc[-1]) else 'fail'
                if exp_test_result == 'pass':
                    ora_record[0] += 1
                elif exp_test_result == 'fail':
                    ora_record[1] += 1

                strNumber = bin(number)[2:]
                strNumber = strNumber.zfill(n)  
                initial_state = [int(bit) for bit in strNumber]
                
                qc_initial = QuantumCircuit(2 * num_qubits + 1, 1)
                for index, val in enumerate(initial_state[::-1]):
                    if initial_gates[val] == 1:
                        qc_initial.x(index)
                                                    
                # running the quantum programs
                qc = qc_initial.copy()

                # need to vary with different programs
                func = version_selection(program_name, program_version)
                qc_test = func(n, L, sign)
                
                qc_exp = QuantumCircuit(num_qubits)
                exp_value = program_specification_value(n, number, L, sign)
                exp_state = [0] * (2 ** num_qubits)
                exp_state[exp_value] = 1
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
                        
                if exp_test_result == 'pass' and test_result == 'fail':
                    err_record[0] += 1
                elif exp_test_result == 'fail' and test_result == 'pass':
                    err_record[1] += 1    

            durTime = time.time() - start_time            
            time_list += [durTime]
            failure_list += [failures]                  
        time_records.append(time_list)
        fault_records.append(failure_list)
        err_record = err_record / ora_record
        err_records.append(list(err_record))

    return time_records, fault_records, err_records

def testing_with_STSQ(qubit_list, program_version, total_repeats, shots):    
    '''
        This function implements the test process using STSQ (Swap Test on Separate Qubits)
        
        Input variables:
            + qubit_list:           [list]   the list of qubit numbers
            + program_version:      [str]    the buggy version to be tested, including "v1", "v2", "v3", "v4" and "v5"
            + total_repeats:        [int]    the parameter $r$ involved in the paper standing for independent repetitive  
                                             executions of the same test process  
            + shots:                [int]    the configured shots $s$ 
            + statistical_methods:  [str]    the concrete OPO, including "MWTest", "ChiTest", "KSTest", "CrsEnt" 
                                             and "JSDiv"

        Output variables:
            + time_records:         [list]   the execution time for $r$ repeats
            + fault_records:        [list]   the detected faults for $r$ repeats
            + err_records:          [list]   the estimated type I and type II errors
    '''
    time_records = []         
    fault_records = []        
    err_records = []   
    initial_gates = [0, 1]
    
    print('STSQ')

    for num_qubits in tqdm(qubit_list):                   
        filename = '/' + file_short_name + '_' + str(program_version) + '_testSuites_(qubit=' \
            + str(num_qubits) + ',fr=0.5,#t=50).csv'
        df = pd.read_csv(parent_dir + filename, skiprows=0)
        err_record = np.array([0, 0])                   # err_record[0]: type I; err_record[1]: type II
        ora_record = np.array([0, 0])                   # ora_record[0]: "pass"; ora_record[1]: "fail"
        time_list = []
        failure_list = []
        for index in range(total_repeats):     # repeat for statistically significant results
            start_time = time.time()
            failures = 0              
            for test_order in range(len(df)): 
                test_input = df.iloc[test_order]                   
                step = test_order + 1
                
                # need to vary with different programs
                n, num_qubits = test_input.iloc[0], test_input.iloc[1]
                number, L, sign = test_input.iloc[2], test_input.iloc[3], test_input.iloc[4]

                exp_test_result = 'pass' if bool(test_input.iloc[-1]) else 'fail'
                if exp_test_result == 'pass':
                    ora_record[0] += 1
                elif exp_test_result == 'fail':
                    ora_record[1] += 1

                # transform number to binary list
                strNumber = bin(number)[2:]
                strNumber = strNumber.zfill(n) 
                initial_state = [int(bit) for bit in strNumber]

                qc_initial = QuantumCircuit(num_qubits + 2, 1)
                for index, val in enumerate(initial_state[::-1]):
                    if initial_gates[val] == 1:
                        qc_initial.x(index)
                
                # select a qubit to test 
                qubitTestList = np.random.choice(range(num_qubits), 
                                                size=num_qubits, 
                                                replace=False)
                
                test_result = 'pass'
                for selOrder, tempQubit in enumerate(qubitTestList):                          
                    # running the quantum programs
                    qc = qc_initial.copy()
                    
                    # need to vary with different programs
                    func = version_selection(program_name, program_version)
                    qc_test = func(n, L, sign)

                    qc.append(qc_test, qc.qubits[:num_qubits])
                    beta, theta = program_specification_angles(tempQubit, n, number, L, sign)
                    qc.ry(theta, -2)
                    qc.rz(beta, -2)
                    qc.h(-1)
                    qc.cswap(-1, tempQubit, -2)
                    qc.h(-1)
                    qc.measure(qc.qubits[-1], qc.clbits)

                    dict_counts = circuit_execution(qc, shots)
            
                    # transform a dict into a list
                    resList = list(dict_counts.keys())
                    
                    # step += 1
                    if len(resList) == 1 and resList[0] == 0:   # this qubit passes                     
                        continue
                            
                    else:                                       # this qubit fails
                        test_result = 'fail'
                        break

                failures += int(test_result == 'fail')   

                if exp_test_result == 'pass' and test_result == 'fail':
                    err_record[0] += 1
                elif exp_test_result == 'fail' and test_result == 'pass':
                    err_record[1] += 1    

            durTime = time.time() - start_time            
            time_list += [durTime]
            failure_list += [failures]                  
        time_records.append(time_list)
        fault_records.append(failure_list)
        err_record = err_record / ora_record
        err_records.append(list(err_record))
    
    return time_records, fault_records, err_records

def testing_with_HOSS(qubit_list, program_version, total_repeats, shots):
    '''
        This function implements the test process using HOSS (Hybrid Oracle via Separable States)
        
        Input variables:
            + qubit_list:           [list]   the list of qubit numbers
            + program_version:      [str]    the buggy version to be tested, including "v1", "v2", "v3", "v4" and "v5"
            + total_repeats:        [int]    the parameter $r$ involved in the paper standing for independent repetitive  
                                             executions of the same test process  
            + shots:                [int]    the configured shots $s$ 
            + statistical_methods:  [str]    the concrete OPO, including "MWTest", "ChiTest", "KSTest", "CrsEnt" 
                                             and "JSDiv"

        Output variables:
            + time_records:         [list]   the execution time for $r$ repeats
            + fault_records:        [list]   the detected faults for $r$ repeats
            + err_records:          [list]   the estimated type I and type II errors
    '''
    time_records = []         
    fault_records = []        
    err_records = [] 
    initial_gates = [0, 1]
    
    print("HOSS")

    for num_qubits in tqdm(qubit_list):                   
        filename = '/' + file_short_name + '_' + str(program_version) + '_testSuites_(qubit=' \
            + str(num_qubits) + ',fr=0.5,#t=50).csv'
        df = pd.read_csv(parent_dir + filename, skiprows=0)
        err_record = np.array([0, 0])                   # err_record[0]: type I; err_record[1]: type II
        ora_record = np.array([0, 0])                   # ora_record[0]: "pass"; ora_record[1]: "fail"
        time_list = []
        failure_list = []
        for index in range(total_repeats):     # repeat for statistically significant results
            start_time = time.time()
            failures = 0
            for test_order in range(len(df)): 
                test_input = df.iloc[test_order]                   
                
                # need to vary with different programs
                n, num_qubits = test_input.iloc[0], test_input.iloc[1]
                number, L, sign = test_input.iloc[2], test_input.iloc[3], test_input.iloc[4]
                
                exp_test_result = 'pass' if bool(test_input.iloc[-1]) else 'fail'
                if exp_test_result == 'pass':
                    ora_record[0] += 1
                elif exp_test_result == 'fail':
                    ora_record[1] += 1

                strNumber = bin(number)[2:]
                strNumber = strNumber.zfill(n)  
                initial_state = [int(bit) for bit in strNumber]
                
                qc_initial = QuantumCircuit(num_qubits, num_qubits)
                for index, val in enumerate(initial_state[::-1]):
                    if initial_gates[val] == 1:
                        qc_initial.x(index)
                            
                # running the quantum programs
                qc = qc_initial.copy()

                # need to vary with different programs
                func = version_selection(program_name, program_version)
                qc_test = func(n, L, sign)

                qc.append(qc_test, qc.qubits)
                qc.measure(qc.qubits,qc.clbits)
                    
                dict_counts = circuit_execution(qc, shots)
                
                # transform a dict into a list
                exp_value = program_specification_value(n, number, L, sign)

                resList = list(dict_counts.keys())
                if len(resList) == 1 and resList[0] == exp_value:
                    test_result = 'pass'
                else:
                    test_result = 'fail'

                failures += int(test_result == 'fail')                   
                if exp_test_result == 'pass' and test_result == 'fail':
                    err_record[0] += 1
                elif exp_test_result == 'fail' and test_result == 'pass':
                    err_record[1] += 1    

            durTime = time.time() - start_time            
            time_list += [durTime]
            failure_list += [failures]                  
        time_records.append(time_list)
        fault_records.append(failure_list)
        err_record = err_record / ora_record
        err_records.append(list(err_record))
    return time_records, fault_records, err_records