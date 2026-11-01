from collections import Counter
import struct
import random
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(BASE_DIR))

import model

EMBEDDING_SIZE = model.embedding_size()

SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<USER>", "<TUL>", " "]

NUM_MERGES = 100

TRAIN_PATH = BASE_DIR / "files" / "train.md"
MERGES_PATH = BASE_DIR / "data" / "merges.md"
TOKENS_PATH = BASE_DIR / "data" / "tokens.md"
WEIGHTS_PATH = BASE_DIR / "data" / "weights" / "weights.bin"


def unpack_weights(path=WEIGHTS_PATH, embedding_size=EMBEDDING_SIZE):
    """Read weights.bin and return one list of `embedding_size` floats per token.

    Row i is the embedding for the token on line i of tokens.md.
    """
    data = Path(path).read_bytes()

    row_bytes = embedding_size * 4  # float32 = 4 bytes

    if len(data) % row_bytes != 0:
        raise ValueError(
            f"{path} has {len(data)} bytes, which isn't a multiple of "
            f"{row_bytes} (embedding size {embedding_size} x 4 bytes)."
        )

    flat = struct.unpack(f"<{len(data) // 4}f", data)

    return [list(flat[i:i + embedding_size]) for i in range(0, len(flat), embedding_size)]


def load_tokens(path=TOKENS_PATH):
    """Return the token list; a token's ID is its index."""
    with open(path, "r", encoding="utf-8") as file:
        return file.read().splitlines()


def main():
    print("Getting Merges...")

    with open(TRAIN_PATH, "r", encoding="utf-8") as file:
        train_text = file.read()

    words = train_text.split()

    words = [word for word in words if word not in SPECIAL_TOKENS]

    words = [list(word) for word in words]

    # Record every base character before merging mutates `words`
    base_chars = {ch for word in words for ch in word}

    learned_merges = []

    for step in range(NUM_MERGES):

        pairs = []

        for word in words:
            pairs.extend(zip(word, word[1:]))

        counts = Counter(pairs)

        if not counts:
            break

        pair, frequency = counts.most_common(1)[0]

        if frequency < 2:
            break

        merged = pair[0] + pair[1]

        learned_merges.append(pair)

        for word in words:
            i = 0

            while i < len(word) - 1:
                if (word[i], word[i + 1]) == pair:
                    word[i] = merged
                    word.pop(i + 1)
                else:
                    i += 1

    MERGES_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(MERGES_PATH, "w", encoding="utf-8") as file:
        for pair in learned_merges:
            file.write(f"{pair[0]} {pair[1]}\n")

    print(f"{len(learned_merges)} merges successfully stored inside of data/merges.md.")

    print("Getting Tokens...")

    # Special tokens first (so <PAD> is ID 0), then base characters,
    # then merged tokens in the order they were learned
    merged_tokens = [a + b for a, b in learned_merges]
    ordered_vocab = list(dict.fromkeys(list(SPECIAL_TOKENS) + sorted(base_chars) + merged_tokens))

    existing_tokens = set(load_tokens()) if TOKENS_PATH.exists() else set()

    new_tokens = [token for token in ordered_vocab if token not in existing_tokens]

    with open(TOKENS_PATH, "a", encoding="utf-8") as file:
        for token in new_tokens:
            file.write(token + "\n")

    print(f"Added {len(new_tokens)} new tokens.")

    print("Getting Weights...")

    new_weights = []

    for token in new_tokens:
        for _ in range(EMBEDDING_SIZE):
            new_weights.append(random.uniform(-0.1, 0.1))

    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(WEIGHTS_PATH, "ab") as file:
        if new_weights:
            file.write(struct.pack(f"<{len(new_weights)}f", *new_weights))

    print("Weights successfully stored.")

    # Sanity check: one row of EMBEDDING_SIZE float32s per token
    total_tokens = len(load_tokens())

    expected_bytes = total_tokens * EMBEDDING_SIZE * 4
    actual_bytes = WEIGHTS_PATH.stat().st_size

    if actual_bytes != expected_bytes:
        print(f"WARNING: weights.bin has {actual_bytes} bytes, expected {expected_bytes} "
              f"for {total_tokens} tokens. Tokens and weights are out of sync.")
    else:
        print(f"Vocabulary size: {total_tokens} tokens, weights in sync.")

    print(f"Tokens: {total_tokens} | Parameters: {total_tokens * EMBEDDING_SIZE}")


if __name__ == "__main__":
    main()