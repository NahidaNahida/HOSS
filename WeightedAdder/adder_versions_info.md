# WA - v1
    bug:
        line 257: delete a \texttt{cx} gate
        - circuit.cx(q_state, qr_sum[j])         

    complexity:
        # qubits = 6 :  # gates = [0, 144],     depth = [0, 94]
        # qubits = 7 :  # gates = [15, 144],    depth = [11, 94]
        # qubits = 8 :  # gates = [15, 144],    depth = [11, 94]
        # qubits = 9 :  # gates = [15, 144],    depth = [11, 94]
        # qubits = 10:  # gates = [15, 144],    depth = [11, 94]
        # qubits = 11:  # gates = [15, 144],    depth = [11, 94]
        # qubits = 12:  # gates = [15, 144],    depth = [11, 94]

# WA - v2 
    bug:
        line 297: reverse a process
        - for j in reversed(range(len(weight_binary))):
        + for j in (range(len(weight_binary))):   # bug

        line 303: delete an \texttt{x} gate
        - circuit.x(qr_sum[j])  
    
    complexity:
        # qubits = 10:  # gates = [235, 240],  depth = [72, 72]

# WA - v3
    bug:
        line 303: replace the \texttt{x} gate with an \texttt{h} gate
        - circuit.x(qr_sum[j])
        + circuit.h(qr_sum[j]) # bug

        line 305: replace the \texttt{x} gate with an \texttt{h} gate
        - circuit.x(qr_sum[j])
        + circuit.h(qr_sum[j]) # bug

    complexity:
        # qubits = 10:  # gates = [250, 255],  depth = [75, 75]

# WA - v4 
    bug:
        line 304: switch qubits of the \texttt{ccx} gate
        - circuit.ccx(q_state, qr_sum[j], qr_carry[j])
        + circuit.ccx(q_state, qr_carry[j], qr_sum[j]) # bug

    complexity: 
        # qubits = 10:  # gates = [265, 270],  depth = [78, 78]

# WA - v5
    bug:
        line 252: replace the \texttt{cx} gate with a \texttt{ch} gate
        - circuit.cx(q_state, qr_sum[j])
        + circuit.ch(q_state, qr_sum[j])  # def

        line 257: replace the \texttt{cx} gate with a \texttt{ch} gate
        - circuit.cx(q_state, qr_sum[j])
        + circuit.ch(q_state, qr_sum[j])  # def

    complexity:
        # qubits = 10:  # gates = [250, 255],  depth = [75, 75]

