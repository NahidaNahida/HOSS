# IC - v1
    bug:
        line 233: replace the \texttt{cx} gate with a \texttt{ch} gate
        - circuit.cx(qr_state[i], qr_ancilla[i])
        + circuit.ch(qr_state[i], qr_ancilla[i]) # bug

    complexity:
        # qubits = 6 :  # gates = [0, 69] ,  depth = [0, 43]
        # qubits = 8 :  # gates = [0, 109],  depth = [0, 66]
        # qubits = 10:  # gates = [0, 149],  depth = [0, 89]
        # qubits = 12:  # gates = [0, 189],  depth = [0, 112]

# IC - v2
    bug:
        line 227: replace the \texttt{x} gate with an \texttt{h} gate
        - circuit.x(q_compare)
        + circuit.h(q_compare)                             # x -> h

    complexity:
        # qubits = 10:  # gates = [0, 143],  depth = [0, 83]

# IC - v3
    bug:
        line 240: switch qubits in a \texttt{ccx} gate
        - circuit.ccx(qr_state[i], qr_ancilla[i - 1], qr_ancilla[i])
        + circuit.ccx(qr_ancilla[i], qr_ancilla[i - 1], qr_state[i])      # defect switch qubit

    complexity:
        # qubits = 10:  # gates = [0, 143],  depth = [0, 83]

# IC - v4
    bug:
        line 206: delete the \texttt{if} condition
        - if twos[i] == 1:
        
        line 207: replace the \texttt{cx} gate with a \texttt{ry} gate
        - circuit.cx(qr_state[i], qr_ancilla[i])
        + circuit.ry(3.141592/5, qr_state[i])                     #  replace gate

        line 208: add a \texttt{crz} gate
        + circuit.crz(3.141592/3, qr_state[i], qr_ancilla[i])     #  add gate

    complexity:
        # qubits = 10:  # gates = [0, 147],  depth = [0, 86]

# IC - v5
    bug:
        line 241: add an \texttt{h} gate
        + circuit.h(qr_state) # bug

    complexity:
        # qubits = 10:  # gates = [0, 142],  depth = [0, 83]
