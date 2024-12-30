# DO - v1
    bug:
        line 110: add a \texttt{ch} gate
        + circuit.ch(0, -1)       # defect: add

    complexity:
        # qubits = 6:   # gates = [93, 105],    depth = [77, 88]
        # qubits = 7:   # gates = [162, 224],   depth = [140, 201]
        # qubits = 8:   # gates = [299, 359],   depth = [268, 328]
        # qubits = 9:   # gates = [562, 622],   depth = [524, 584]
        # qubits = 10:  # gates = [1081, 1333], depth = [1035, 1287]  
        # qubits = 11:  # gates = [2112, 2364], depth = [2058, 2310]
        # qubits = 12:  # gates = [4167, 4419], depth = [4105, 4354]

# DO - v2 
    bug:
        line 102: modify the condition of \texttt{while}
        - while n >= 2:      
        + while n >= 8:        # >= 2 -> >= 8

    complexity:
        # qubits = 10:  # gates = [1024, 1275], depth = [1017, 1268]

# DO - v3
    bug:
        line 105: switch two inputs of a subroutine
        - diag_phases[i // 2], rz_angle = _extract_rz(diag_phases[i], diag_phases[i + 1])
        + diag_phases[i // 2], rz_angle = _extract_rz(diag_phases[i + 1], diag_phases[i]) # def: switch diag_phases[i], diag_phases[i + 1]

    complexity:
        # qubits = 10:  # gates = [1027, 1087], depth = [1019, 1080]

# DO - v4 
    bug:
        line 102: add a \texttt{cx} gate
        + circuit.cx(0, 1)        # add gate

    complexity: 
        # qubits = 10:  # gates = [1025, 1278], depth = [1018, 1265]

# DO - v5
    bug:
        line 110: modify the target qubit of a \texttt{ucrz} gate
        - circuit.ucrz(angles_rz, ctrl_qubits, target_qubit)
        + circuit.ucrz(angles_rz, ctrl_qubits, 0) 

    complexity:
        # qubits = 10:  # gates = [1027, 1279], depth = [1027, 1279]