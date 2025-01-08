import os
import pandas as pd
import csv
from amplitude_test_process_RQ3 import *

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)    
OPOs = ['ChiTest', 'KSTest', 'MWTest', 'JSDiv', 'CrsEnt']
WOOs = ['STFQ', 'STSQ', 'HOSS']

def RQ3_running(test_oracles, shots, num_qubits, total_repeats=20):
    version_list = ['v1']          
    for program_version in version_list:
        print(program_version)

        for test_oracle in test_oracles:
            if test_oracle in OPOs:
                time_records, fault_records, error_records = testing_with_statistical_methods(num_qubits, 
                                                                                              program_version, 
                                                                                              total_repeats, 
                                                                                              shots, 
                                                                                              test_oracle)
            elif test_oracle == 'STFQ':
                time_records, fault_records, error_records = testing_with_STFQ(num_qubits, 
                                                                               program_version, 
                                                                               total_repeats, 
                                                                               shots)
            elif test_oracle == 'STSQ':
                time_records, fault_records, error_records = testing_with_STSQ(num_qubits, 
                                                                               program_version, 
                                                                               total_repeats, 
                                                                               shots)
            elif test_oracle == 'HOSS':
                time_records, fault_records, error_records = testing_with_HOSS(num_qubits, 
                                                                               program_version, 
                                                                               total_repeats, 
                                                                               shots)
            
            file_name = "LAF_" + str(program_version) + "_RQ3_" + str(test_oracle) + ".csv"
            with open(file_name, mode='w', newline='') as file:
                writer = csv.writer(file)
                header = ['# qubits', 'time_list', 'fault_list', 'ave_time', 'ave_fault', 
                          'std_time', 'std_fault', 'Type I', 'Type II']
                writer.writerow(header)
                for index, num_qubits in enumerate(list(num_qubits)):
                    time_list, fault_list = time_records[index], fault_records[index]
                    err_type1, err_type2 = error_records[index][0], error_records[index][1]
                    data = [num_qubits, list(time_list), list(fault_list),
                            np.mean(time_list), np.mean(fault_list),
                            np.std(time_list, ddof=1), np.std(fault_list, ddof=1),
                            err_type1, err_type2]
                    writer.writerow(data)

if __name__ == '__main__':
    test_oracles = OPOs + WOOs
    num_qubits = [6, 7, 8, 9, 10, 11, 12]
    shots = 10   
    RQ3_running(test_oracles, shots, num_qubits)