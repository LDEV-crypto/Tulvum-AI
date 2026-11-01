import re
import sys
from pathlib import Path

import torch
from torch import nn

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from model import EMBEDDING_SIZE, WEIGHTS_FILE, attention_heads, find_tokens, load_pretrained_embeddings, load_tokens, token_to_id
from transformer.self_attention import TransformerBlock

TRAIN_PATH = BASE_DIR / "files" / "train.md"
VALIDATION_PATH = BASE_DIR / "files" / "validation.md"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
SEQUENCE_LENGTH = 12


class TinyLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, pretrained_path=WEIGHTS_FILE):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.position = nn.Embedding(SEQUENCE_LENGTH + 4, embed_dim)
        self.block = TransformerBlock(embed_dim, attention_heads(), embed_dim * 2)
        self.output = nn.Linear(embed_dim, vocab_size)
        self._load_pretrained(pretrained_path)

    def _load_pretrained(self, pretrained_path):
        vectors = load_pretrained_embeddings(pretrained_path, embedding_size=self.embedding.embedding_dim, vocab_size=self.embedding.num_embeddings)
        if vectors is None:
            return

        weights = torch.tensor(vectors, dtype=torch.float32)
        if weights.shape[0] > self.embedding.num_embeddings:
            weights = weights[:self.embedding.num_embeddings]
        if weights.shape[0] < self.embedding.num_embeddings:
            pad = torch.zeros(self.embedding.num_embeddings - weights.shape[0], self.embedding.embedding_dim, dtype=torch.float32)
            weights = torch.cat([weights, pad], dim=0)
        if weights.shape == self.embedding.weight.shape:
            with torch.no_grad():
                self.embedding.weight.copy_(weights)

    def forward(self, token_ids):
        single = token_ids.dim() == 1
        if single:
            token_ids = token_ids.unsqueeze(0)
        positions = torch.arange(token_ids.size(1), device=token_ids.device).unsqueeze(0)
        x = self.embedding(token_ids) + self.position(positions)
        x = self.block(x)
        x = self.output(x)
        if single:
            return x[0]
        return x

    def predict(self, token_ids):
        logits = self(token_ids)
        return torch.argmax(logits, dim=-1)


def build_vocabulary():
    return {token: index for index, token in enumerate(load_tokens())}


def encode_text(text):
    cleaned = text.replace("\n", " ")
    pieces = []
    for chunk in re.findall(r"<[^>]+>|\S+|\s+", cleaned):
        chunk = chunk.strip()
        if not chunk:
            continue
        if chunk.startswith("<") and chunk.endswith(">"):
            pieces.append(chunk)
        else:
            for token in find_tokens(chunk):
                pieces.append(token)
    return pieces


def build_text_samples(path):
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="ignore")
    return encode_text(text)


def build_training_data(sequence_length=SEQUENCE_LENGTH, limit=2000):
    vocab = build_vocabulary()
    tokens = []
    for path in (TRAIN_PATH, VALIDATION_PATH):
        tokens.extend(build_text_samples(path))
    if not tokens:
        tokens = ["<BOS>", "<EOS>"]
    ids = [token_to_id(token, vocab.get("<UNK>", 0)) for token in tokens]
    samples = []
    for start in range(0, max(1, len(ids) - 1), max(1, sequence_length // 2)):
        window = ids[start:start + sequence_length]
        if len(window) < 2:
            continue
        inputs = torch.tensor(window[:-1], dtype=torch.long)
        targets = torch.tensor(window[1:], dtype=torch.long)
        samples.append((inputs, targets))
        if len(samples) >= limit:
            break
    if not samples:
        base = [vocab.get("<BOS>", 0), vocab.get("<EOS>", 1)]
        samples = [(torch.tensor(base[:-1], dtype=torch.long), torch.tensor(base[1:], dtype=torch.long))]
    return samples


def build_validation_data(sequence_length=SEQUENCE_LENGTH, limit=500):
    vocab = build_vocabulary()
    tokens = build_text_samples(VALIDATION_PATH)
    if not tokens:
        return build_training_data(sequence_length=sequence_length, limit=limit)
    ids = [token_to_id(token, vocab.get("<UNK>", 0)) for token in tokens]
    samples = []
    for start in range(0, max(1, len(ids) - 1), max(1, sequence_length // 2)):
        window = ids[start:start + sequence_length]
        if len(window) < 2:
            continue
        samples.append((torch.tensor(window[:-1], dtype=torch.long), torch.tensor(window[1:], dtype=torch.long)))
        if len(samples) >= limit:
            break
    return samples if samples else build_training_data(sequence_length=sequence_length, limit=limit)


def batch_data(data, batch_size=8):
    batches = []
    current = []
    for item in data:
        current.append(item)
        if len(current) == batch_size:
            batches.append(current)
            current = []
    if current:
        batches.append(current)
    return batches


def train_epoch(model, data, learning_rate=1e-3):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_loss = 0.0
    batches = batch_data(data)
    for batch in batches:
        optimizer.zero_grad()
        loss_value = 0.0
        for inputs, targets in batch:
            logits = model(inputs)
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss_value += loss.item()
            loss = loss / len(batch)
            loss.backward()
        optimizer.step()
        total_loss += loss_value / max(len(batch), 1)
    return total_loss / max(len(batches), 1)


def save_checkpoint(model, path):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    state = {"model": model.state_dict()}
    torch.save(state, path)


def main():
    torch.manual_seed(42)
    vocab = load_tokens()
    dataset = build_training_data()
    validation = build_validation_data()
    model = TinyLanguageModel(len(vocab), EMBEDDING_SIZE)
    for epoch in range(3):
        loss = train_epoch(model, dataset)
        val_loss = train_epoch(model, validation)
        print(f"epoch {epoch + 1} loss {loss:.4f} validation {val_loss:.4f}")
    save_checkpoint(model, CHECKPOINT_DIR / "training_state.pt")
    print("grammar and speech training pipeline complete")


if __name__ == "__main__":
    main()
