from qiskit import transpile
from qiskit_aer import Aer

def circuit_execution(qc, shots):# the actual test process
    """
        Execute the quantum circuit with given shots, and then return the measurement results.
    """     
    backend = Aer.get_backend('qasm_simulator')
    executed_qc = transpile(qc, backend)
    count= backend.run(executed_qc, shots=shots).result().get_counts()
    dict_counts = count.int_outcomes()
    return dict_counts

    
def circuit_execution_fake(qc):
    """
        Execute the quantum circuit with given shots, and then return the measurement results.
    """
    backend = Aer.get_backend('statevector_simulator')
    executed_qc = transpile(qc, backend)
    result = backend.run(executed_qc).result()
    statevector = result.get_statevector()
    return statevector