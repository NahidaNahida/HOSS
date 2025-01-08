from qiskit import QuantumCircuit 
import numpy as np
import csv
from tqdm import tqdm

from adder import WeightedAdder
from adder_defect1 import WeightedAdder_defect1
from adder_defect2 import WeightedAdder_defect2
from adder_defect3 import WeightedAdder_defect3
from adder_defect4 import WeightedAdder_defect4
from adder_defect5 import WeightedAdder_defect5

from preprossing import number_list_generation

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)  

from circuit_execution import *

def version_selection(program_name, program_version):
    '''
        Select the quantum program to be tested

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

def test_input_generation(n_list, totalQubits, maxTests=None):
    '''
        Generation the test inputs
    
        Input variables:
            + n_list             [list]  the list of qubit numbers
                                         note that n only corresponds to the input domain
            + maxTests           [int]   the maximum size of the test suite

        Output variable:
            + testOrderListRaw   [list]  the randomized order for executing test cases
            + testInput          [list]  the list of test inputs
    '''
    # the maximum number of test cases
    initial_gates = [0, 1]
    candiWeights = [0, 1, 2, 3]
    tempNum = 0
    testInput = []

    # generate A random integer matrix A and integer vector b
    for n in n_list:
        weightList = number_list_generation(n, len(candiWeights))
        initial_states = number_list_generation(n, len(initial_gates))
        for initial_state in initial_states:
            for weight in weightList:                
                qc = WeightedAdder(n, weight)
                numQubits = qc.num_qubits
                if totalQubits == numQubits:
                    tempNum += 1
                    print(tempNum)
                    testInput.append([n, initial_state, numQubits, weight])
                    if maxTests != None and tempNum >= maxTests:
                        numTests = len(testInput)
                        maxTests = numTests if maxTests == None else maxTests
                        testOrderListRaw = np.random.choice(range(numTests), size=min(numTests, maxTests), replace=False)
                        return testOrderListRaw, testInput

def program_running(testOrderListRaw, testInput, version, maxTests=None, faultRatio=0.5, eps_0=1e-10):
    '''
        determine the ground truth, select test cases with a required distribution of faults, and output the
        test suites

        Variables:
            + testOrderListRaw   [list]  the randomized order for executing test cases
            + testInput          [list]  the list of test inputs
            + version            [str]   the program version, e.g., 'v1', 'v2', etc.
            + maxTests           [int]   the required size of the generated test suite
            + faultRation        [float] the proportion of failed test cases determined by the ground truth
            + eps_0              [float] the threshold to identify the failure of test cases
    '''
 

    initial_gates = [0, 1]
    faultTestSuite = []
    correTestSuite = []
    
    numTests = len(testOrderListRaw)
    maxTests = numTests if maxTests == None else min(maxTests, numTests)

    testOrderList = np.random.choice(testOrderListRaw, size=numTests, replace=False)
    for testOrder in tqdm(testOrderList):                    
        initial_state = testInput[testOrder][1]
        numQubits = testInput[testOrder][2]
        weight = testInput[testOrder][3]
        
        n = len(initial_state)
        number = int(''.join(map(str, initial_state)), 2)   # the decimal number
        
        qc_initial = QuantumCircuit(numQubits)
        
        for index, val in enumerate(initial_state[::-1]):
            if initial_gates[val] == 1:
                qc_initial.x(index)
                    
        # running the quantum programs
        qc_res = qc_initial.copy()
        qc_exp = qc_initial.copy()
        func = version_selection("WeightedAdder", version)
        qc_test = func(n, weight)
        qc_raw = WeightedAdder(n, weight)
        qc_res.append(qc_test, qc_res.qubits)
        qc_exp.append(qc_raw, qc_exp.qubits)
        
        # derive the state vector to analyze whether the test really fails or not
        test_vec = circuit_execution_fake(qc_res)
        exp_vec = circuit_execution_fake(qc_exp)
        fidelity = (np.abs(np.vdot(np.array(test_vec), np.array(exp_vec)))) ** 2
        truth =  bool(fidelity >= 1 - eps_0)
        
        test_case = [n, qc_exp.num_qubits, number, weight, truth]
                  
        if truth:  
            correTestSuite.append(test_case)
        else:
            faultTestSuite.append(test_case)
            
    num_faults = int(round(maxTests * faultRatio))
    num_faults_raw = len(faultTestSuite)
    num_correct_raw = len(correTestSuite)
    if num_faults_raw >= num_faults:
        faultTestSuiteIndex = np.random.choice(range(num_faults_raw), size=num_faults, replace=False)
    else:
        faultTestSuiteIndex = np.random.choice(range(num_faults_raw), size=num_faults_raw, replace=False)
    correTestSuiteIndex = np.random.choice(range(num_correct_raw), size=maxTests-num_faults,replace=False)
    
    faultTestSuiteOpt = [faultTestSuite[index] for index in faultTestSuiteIndex]
    correTestSuiteOpt = [correTestSuite[index] for index in correTestSuiteIndex]
    print("# faults = {}".format(len(faultTestSuiteOpt)))
    print("# correc = {}".format(len(correTestSuiteOpt)))
    finalTestSuite = correTestSuiteOpt + faultTestSuiteOpt
    count_tests = len(faultTestSuiteOpt)+len(correTestSuiteOpt)

    file_name = "WA_{}_testSuites_(qubit={},fr={},#t={}).csv".format(str(version), 
                                                                     str(qc_exp.num_qubits),
                                                                     str(faultRatio),
                                                                     str(count_tests))

    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ['n', '# qubits','number', 'weight', 'if_pass']
        writer.writerow(header)

        for tempTest in finalTestSuite:
            writer.writerow(tempTest)
    print('done!')

if __name__ == '__main__':
    n_list = [7, 8, 9]
    total = 10
    versionList = ['v2']
    testOrderListRaw, testInput = test_input_generation(n_list, total, 1000)
    for version in versionList:
        program_running(testOrderListRaw, testInput, version, maxTests=50, faultRatio=0.5)
    