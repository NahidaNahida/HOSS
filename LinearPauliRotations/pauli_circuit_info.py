import pandas as pd
import sys, os
import ast

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)   

from statistical_tests import *
from preprossing import *
from circuit_execution import *

from pauli_defect1 import LinearPauliRotations_defect1
from pauli_defect2 import LinearPauliRotations_defect2
from pauli_defect3 import LinearPauliRotations_defect3
from pauli_defect4 import LinearPauliRotations_defect4
from pauli_defect5 import LinearPauliRotations_defect5

program_name = 'LinearPauliRotations'

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
    
def fully_decompose(qc):
    '''
        thoroughly decompose a qc until only basis gates
    '''
    while True:
        decomposed_qc = qc.decompose()
        if decomposed_qc == qc:
            break
        qc = decomposed_qc
    return qc

def info_collection(num_qubits, program_version):
    '''
        count the gate numbers and depths corresponding to the tested program and print the data
        
        input variables:
        + num_qubits        [int]   the number of qubits involved in the quantum circuit
        + program_version   [str]   the version of the tested program, e.g., 'v1', 'v2'
    '''
    gates_list = []
    depth_list = []

    filename = '/LPR_' + str(program_version) + '_testSuites_(qubit=' + str(num_qubits) + ',fr=0.5,#t=50).csv'
    df = pd.read_csv(parent_dir + filename, skiprows=0)

    for test_order in range(len(df)): 
        test_input = df.iloc[test_order]                   
        n, num_qubits = test_input.iloc[0], test_input.iloc[1]
        number, slop, offset, basis = test_input.iloc[2], test_input.iloc[3], test_input.iloc[4], test_input.iloc[5]
 
        # running the quantum programs
        func = version_selection(program_name, program_version)
        qc = func(n, slop, offset, basis)

        # counts information of quantum circuits
        decomposed_qc = fully_decompose(qc)
        depths = decomposed_qc.depth()
        gates_dict = decomposed_qc.count_ops()
        gates = sum(gates_dict.values())
        gates_list.append(gates)
        depth_list.append(depths)
    
    print('# gates = [{}, {}]'.format(min(gates_list), max(gates_list)))
    print('depth = [{}, {}]'.format(min(depth_list), max(depth_list)))

if __name__ == "__main__":
    RQ_checks = 'RQ3'      # options: 'RQ1','RQ2','RQ3'

    if RQ_checks in ['RQ1', 'RQ2']:
        num_qubits = 10
        program_versions = ['v1', 'v2', 'v3', 'v4', 'v5']
        for program_version in program_versions:
            print('version = {}, # qubits = {}'.format(program_version, num_qubits))
            info_collection(num_qubits, program_version)
    elif RQ_checks == 'RQ3':
        num_qubits_list = [6, 7, 8, 9, 10, 11, 12]
        program_version = 'v1'
        for num_qubits in num_qubits_list:
            print('version = {}, # qubits = {}'.format(program_version, num_qubits))
            info_collection(num_qubits, program_version)