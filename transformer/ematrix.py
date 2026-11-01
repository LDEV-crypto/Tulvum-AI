import random
import struct
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(BASE_DIR))

import model

EMBEDDING_SIZE = model.embedding_size()
TOKENS = model.tokens()

matrix_file = BASE_DIR / "data" / "weights" / "matrix.bin"

print("Making Embedding Matrix...")

matrix = []

try:
    with open(matrix_file, "rb") as file:
        current_matrix = file.read()

except FileNotFoundError:
    current_matrix = b""

if len(current_matrix) % (EMBEDDING_SIZE * 4) != 0:
    raise ValueError("Existing matrix has an invalid size!")

existing_rows = len(current_matrix) // (EMBEDDING_SIZE * 4)

if existing_rows > TOKENS:
    raise ValueError("Existing matrix has more rows than the vocabulary!")

if existing_rows > 0:
    values = struct.unpack(
        f"<{existing_rows * EMBEDDING_SIZE}f",
        current_matrix
    )

    for row in range(existing_rows):
        start = row * EMBEDDING_SIZE
        end = start + EMBEDDING_SIZE

        matrix.append(list(values[start:end]))

new_rows = TOKENS - existing_rows

for rows in range(new_rows):
    vector = []

    for columns in range(EMBEDDING_SIZE):
        vector.append(random.uniform(-0.05, 0.05))

    matrix.append(vector)

print("Matrix Created! Storing as .bin...")

if new_rows > 0:
    values = [
        value
        for row in matrix[existing_rows:]
        for value in row
    ]

    with open(matrix_file, "ab") as file:
        file.write(
            struct.pack(f"<{len(values)}f", *values)
        )

def unpack_matrix():
    with open(matrix_file, "rb") as file:
        data = file.read()

    row_bytes = EMBEDDING_SIZE * 4

    if len(data) % row_bytes != 0:
        raise ValueError("Invalid embedding matrix size!")

    rows = len(data) // row_bytes

    values = struct.unpack(
        f"<{rows * EMBEDDING_SIZE}f",
        data
    )

    matrix = []

    for row in range(rows):
        start = row * EMBEDDING_SIZE
        end = start + EMBEDDING_SIZE

        matrix.append(list(values[start:end]))

    return matrix

print("Embedding Matrix Stored at data/weights/matrix.bin!")