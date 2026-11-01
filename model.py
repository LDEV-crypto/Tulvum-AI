import struct
from functools import lru_cache
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
TOKENS_FILE = BASE_DIR / "data" / "tokens.md"
WEIGHTS_FILE = BASE_DIR / "data" / "weights" / "weights.bin"
MATRIX_FILE = BASE_DIR / "data" / "weights" / "matrix.bin"

EMBEDDING_SIZE = 128
TRANSFORMER_LAYER = 2
ATTENTION_HEADS = 4
FEED_FORWARD_SIZE = 512


@lru_cache(maxsize=1)
def load_tokens():
    with open(TOKENS_FILE, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def tokens():
    return list(load_tokens())


def token_to_id(token, default=None):
    token_map = {value: index for index, value in enumerate(load_tokens())}
    if token in token_map:
        return token_map[token]
    if default is not None:
        return default
    return token_map.get("<UNK>", 0)


def id_to_token(token_id):
    token_list = load_tokens()
    return token_list[token_id] if 0 <= token_id < len(token_list) else "<UNK>"


def embedding_size():
    return EMBEDDING_SIZE


def transformer_layer():
    return TRANSFORMER_LAYER


def attention_heads():
    return ATTENTION_HEADS


def feed_forward_size():
    return FEED_FORWARD_SIZE


def token_count():
    return len(load_tokens())


def parameter_count():
    return token_count() * EMBEDDING_SIZE


def grab_token(token_id):
    return id_to_token(token_id)


def grab_token_id(token):
    token_list = load_tokens()
    return token_list.index(token) if token in token_list else -1


def find_tokens(word):
    token_list = load_tokens()
    max_len = max((len(t) for t in token_list), default=1)
    result = []
    i = 0

    while i < len(word):
        matched = False
        for size in range(min(max_len, len(word) - i), 0, -1):
            piece = word[i:i + size]
            if piece in token_list:
                result.append(piece)
                i += size
                matched = True
                break
        if not matched:
            result.append(word[i])
            i += 1

    return result


def load_pretrained_embeddings(path=WEIGHTS_FILE, embedding_size=EMBEDDING_SIZE, vocab_size=None):
    file_path = Path(path)
    if not file_path.exists() or file_path.stat().st_size == 0:
        return None

    raw = file_path.read_bytes()
    row_size = embedding_size * 4
    if len(raw) % row_size != 0:
        return None

    flat = struct.unpack(f"<{len(raw) // 4}f", raw)
    rows = len(flat) // embedding_size
    if rows == 0:
        return None

    limit = vocab_size if vocab_size is not None else rows
    vectors = [list(flat[index * embedding_size:(index + 1) * embedding_size]) for index in range(min(rows, limit))]
    return vectors


if __name__ == "__main__":
    print(find_tokens("<USER> <BOS> What is your salve's name? <EOS>"))