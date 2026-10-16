import numpy as np
from qiskit import QuantumCircuit

ATOL = 1e-10


# general encoding (copied from gemini)
def int_to_custom_binary(n, num_bits=3):
    # Step 1: Shift to 0-indexed
    val = n - 1

    # Step 2: Calculate maximum allowed value for the given bits
    max_val = (1 << num_bits) - 1  # Equivalent to 2**num_bits - 1

    # Step 3: Check bounds and raise error if too high or too low
    if val > max_val:
        raise ValueError(
            f"Input {n} is too high to fit in {num_bits} bits (Max allowed: {max_val + 1})"
        )
    if val < 0:
        raise ValueError("Input must be 1 or greater.")

    # Step 4: Format with dynamic bit width padding
    return f"{val:0{num_bits}b}"


def custom_binary_to_int(binary_str):
    # Step 1: Validate that the string only contains 0s and 1s
    if not all(char in "01" for char in binary_str):
        raise ValueError("Input string must contain only binary digits ('0' or '1').")

    # Step 2: Convert binary string back to a base-10 integer
    val = int(binary_str, 2)

    # Step 3: Shift back up to your 1-indexed format
    return val + 1


def encode(x, N, ket=True):
    # take in list of integers and output the encoded states
    # using zig-zag encoding
    encoded = []
    for val in x:
        encoded.append(int_to_custom_binary(val, N))
    if ket:
        return "|" + "".join(encoded) + ">"
    else:
        return "".join(encoded)


# inverse of the encode function
def decode(bitstring, N):

    chunks = [bitstring[i : i + N] for i in range(0, len(bitstring), N)]
    return [custom_binary_to_int(c) for c in chunks]


def verify_unique_matrix(matrix):
    n = len(matrix)
    if n == 0:
        return 1
    k = int(n**0.5)  # box size (3 for a 9x9 board)

    # Check each row for duplicates
    for row in matrix:
        if len(row) != len(set(row)):
            return 0

    # Check each column for duplicates
    for c in range(n):
        col = [matrix[r][c] for r in range(n)]
        if len(col) != len(set(col)):
            return 0

    # Check each k x k box for duplicates
    for br in range(0, n, k):
        for bc in range(0, n, k):
            box = [matrix[r][c] for r in range(br, br + k) for c in range(bc, bc + k)]
            if len(box) != len(set(box)):
                return 0

    return 1


def find_empty_forbidden_values(board):
    forbidden_values = []
    empty_values = []
    for i in range(len(board)):
        for j in range(len(board)):
            if board[i, j] == 0:
                empty_values.append((i, j))

    # got this part from claude code. I had a weird bug and just gave it my code
    # and it gave me this.
    forbidden_values = {}
    n = len(board)

    k = int(n**0.5)

    for i, (r, c) in enumerate(empty_values):
        row_vals = {int(v) for v in board[r, :] if v != 0}
        col_vals = {int(v) for v in board[:, c] if v != 0}

        br, bc = k * (r // k), k * (c // k)
        box_vals = {int(v) for v in board[br : br + k, bc : bc + k].flatten() if v != 0}

        forbidden_values[i] = row_vals | col_vals | box_vals

        # 4 qubits can hold 1..16, but a 9x9 board only uses 1..9
        if len(board) == 9:
            forbidden_values[i] |= {10, 11, 12, 13, 14, 15, 16}

    return forbidden_values, empty_values


# returns which qubit pairs cannot equal one another
def find_empty_pairs(empty_values, k):
    pairs = []
    for a in range(len(empty_values)):
        for b in range(a + 1, len(empty_values)):
            (r1, c1), (r2, c2) = empty_values[a], empty_values[b]
            if r1 == r2 or c1 == c2 or (r1 // k == r2 // k and c1 // k == c2 // k):
                pairs.append((a, b))
    return pairs


def create_one_oracle_gate(e_register, flag, bitstring):
    qubits = list(e_register) + [flag]
    oracle = QuantumCircuit(qubits, name="U_f")
    for i, bit in enumerate(bitstring):
        if bit == "0":
            oracle.x(e_register[i])
    oracle.mcx(list(e_register), flag)
    for i, bit in enumerate(bitstring):
        if bit == "0":
            oracle.x(e_register[i])
    return oracle.to_gate()


# I added this later (from Claude, was running low on time (sorry)),
# basically runs a series of XOR operations on each pair qubit and then chekcs
# if they are the same using an mcx gate to flip the flag.
def create_pair_gate(N):
    qc = QuantumCircuit(2 * N + 1, name="pair")
    for t in range(N):
        qc.cx(t, N + t)  # b := a XOR b
    qc.mcx(list(range(N, 2 * N)), 2 * N, ctrl_state=0)  # fires if all XOR bits are 0
    for t in range(N):
        qc.cx(t, N + t)  # restore b
    return qc.to_gate()


def create_oracle(qregisters, flags, ancilla, forbidden_values, qbits_per_cell, pairs):
    E = len(forbidden_values)
    oracle_circuit = QuantumCircuit(qregisters, flags, [ancilla])

    # 1. build the comparison gates, but don't apply them yet; keep them in a list
    steps = []
    for cell, forbidden_set in forbidden_values.items():
        e_reg = qregisters[cell * qbits_per_cell : (cell + 1) * qbits_per_cell]
        for value in sorted(forbidden_set):
            bitstring = encode([value], qbits_per_cell, False)
            gate = create_one_oracle_gate(e_reg, flags[cell], bitstring)
            steps.append((gate, list(e_reg) + [flags[cell]]))

    pair_gate = create_pair_gate(qbits_per_cell)  # NEW block
    for p, (a, b) in enumerate(pairs):
        a_reg = qregisters[a * qbits_per_cell : (a + 1) * qbits_per_cell]
        b_reg = qregisters[b * qbits_per_cell : (b + 1) * qbits_per_cell]
        steps.append((pair_gate, list(a_reg) + list(b_reg) + [flags[E + p]]))
    # 2. compute: flags start at 1, each clash flips its cell's flag to 0
    oracle_circuit.x(flags)
    for gate, qubits in steps:
        oracle_circuit.append(gate, qubits)

    # 3. the actual answer: fires only if every flag is 1 (every cell valid)
    oracle_circuit.mcx(list(flags), ancilla)

    # We have to uncompute to disentangle the flag and encoding qubits
    # So that the implicit measurement doesn't mess up the diffusion
    # 4. uncompute: the same steps in reverse order
    for gate, qubits in reversed(steps):
        oracle_circuit.append(gate, qubits)
    oracle_circuit.x(flags)

    return oracle_circuit.to_gate()


def diffuser(e_vals, flags, ancilla):
    d = QuantumCircuit(e_vals, flags, [ancilla], name="diffuser")
    d.h(e_vals)
    d.mcx(list(e_vals), ancilla, ctrl_state=0)  # open controls: fires on |00...0>
    d.h(e_vals)
    return d.to_gate()


def success_fun(k, N):
    theta = np.asin(1 / np.sqrt(N))
    return (np.sin((2 * k + 1) * theta)) ** 2
