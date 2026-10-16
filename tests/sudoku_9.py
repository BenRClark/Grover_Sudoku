import math

import matplotlib.pyplot as plt
import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator

import sudoku_grover.grover_functions as gro


def show_sudoku(M):
    N = len(M)
    n = int(round(N**0.5))
    assert n * n == N and all(len(r) == N for r in M), "Matrix must be n^2 x n^2"

    w = len(str(N))  # width of the largest number
    cell = lambda v: (str(v) if v else ".").rjust(w)
    sep = "+" + "+".join("-" * ((w + 1) * n + 1) for _ in range(n)) + "+"

    for i, row in enumerate(M):
        if i % n == 0:
            print(sep)
        blocks = [
            " " + " ".join(cell(v) for v in row[b * n : (b + 1) * n]) + " "
            for b in range(n)
        ]
        print("|" + "|".join(blocks) + "|")
    print(sep)


def run_grover(board, qbits_per_cell):

    forbidden_values, empty_values = gro.find_empty_forbidden_values(board)

    pairs = gro.find_empty_pairs(empty_values, qbits_per_cell)

    e_vals = QuantumRegister(len(empty_values) * qbits_per_cell, "e")
    flags = QuantumRegister(len(empty_values) + len(pairs), "flags")
    a_registers = QuantumRegister(1, "a")
    circuit = QuantumCircuit(e_vals, flags, a_registers, name="U_f")

    oracle = gro.create_oracle(
        e_vals, flags, a_registers[0], forbidden_values, qbits_per_cell, pairs
    )

    circuit = QuantumCircuit(e_vals, flags, a_registers, name="U_f")

    circuit.h(e_vals)
    circuit.x(a_registers)
    circuit.h(a_registers)

    n = len(e_vals)
    N = 2**n
    iterations = math.floor(math.pi / 4 * math.sqrt(N))

    diff = gro.diffuser(e_vals, flags, a_registers[0])
    for _ in range(iterations):
        circuit.append(oracle, circuit.qubits)  # e qubits + ancilla
        circuit.append(diff, circuit.qubits)

    c = ClassicalRegister(len(e_vals), "c")
    circuit.add_register(c)
    circuit.measure(e_vals, c)

    tqc = transpile(circuit, AerSimulator())
    counts = AerSimulator().run(tqc, shots=5024).result().get_counts()

    best = max(counts, key=counts.get)
    print("Most likely State ", best)

    plot_histogram(counts)

    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
    labels, values = zip(*top)

    plt.figure(figsize=(8, 4))
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("counts")
    plt.show()

    answers = gro.decode(best[::-1], qbits_per_cell)
    for i, a in enumerate(answers):
        board[empty_values[i][0], empty_values[i][1]] = a

    print("classical check of board ", gro.verify_unique_matrix(board))
    print("Grover solution of the sudoku ")
    show_sudoku(board)

    k = np.linspace(0, N, 100)

    y = gro.success_fun(k, 2**n)

    plt.plot(k, y)
    plt.xlabel("Iterations")
    plt.ylabel("Success Probability")
    plt.title("Success probability vs. Iterations")
    plt.show()

    print("Iterations to solve problem ", iterations)


qbits_per_cell = 4

board = np.array(
    [
        [4, 8, 3, 9, 2, 1, 6, 5, 7],
        [9, 6, 7, 3, 4, 5, 8, 2, 1],
        [2, 5, 1, 8, 7, 6, 4, 9, 3],
        [5, 4, 8, 0, 3, 2, 9, 7, 6],
        [7, 2, 9, 5, 6, 4, 1, 3, 8],
        [1, 3, 6, 7, 9, 8, 2, 4, 5],
        [3, 0, 2, 6, 8, 9, 5, 1, 4],
        [8, 0, 4, 2, 5, 3, 7, 6, 9],
        [6, 9, 5, 4, 1, 0, 3, 8, 2],
    ]
)

print("initial board: ")

show_sudoku(board)

run_grover(board, qbits_per_cell)
