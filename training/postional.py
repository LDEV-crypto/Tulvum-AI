import torch
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(BASE_DIR))

tokens_file = BASE_DIR / "data" / "tokens.md"
train_file = BASE_DIR / "files" / "train.md"

import model

EMBEDDING_SIZE = model.embedding_size()

with open(tokens_file, "r") as file:
    tokens = file.read()

with open(tokens_file, "r") as file:
    sentences = file.read()

def positional_encoding(id):
    print(model.grab_token(id))

positional_encoding(0)