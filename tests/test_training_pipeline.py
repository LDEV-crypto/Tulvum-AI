import unittest

import torch

from model import EMBEDDING_SIZE, attention_heads, load_pretrained_embeddings, tokens
from training.train import TinyLanguageModel, build_training_data, train_epoch
from transformer.self_attention import SelfAttention


class TrainingPipelineTests(unittest.TestCase):
    def test_vocab_is_loaded(self):
        vocab = tokens()
        self.assertGreater(len(vocab), 0)
        self.assertIn("<BOS>", vocab)

    def test_pretrained_embeddings_load(self):
        vectors = load_pretrained_embeddings()
        self.assertIsNotNone(vectors)
        self.assertGreater(len(vectors), 0)
        self.assertEqual(len(vectors[0]), EMBEDDING_SIZE)

    def test_self_attention_matches_shape(self):
        layer = SelfAttention(EMBEDDING_SIZE, attention_heads())
        x = torch.randn(2, 5, EMBEDDING_SIZE)
        y = layer(x)
        self.assertEqual(y.shape, x.shape)

    def test_training_data_and_model_are_usable(self):
        data = build_training_data()
        self.assertGreater(len(data), 0)
        model = TinyLanguageModel(len(tokens()), EMBEDDING_SIZE)
        inputs, targets = data[0]
        logits = model(inputs)
        self.assertEqual(logits.shape[-1], len(tokens()))
        self.assertEqual(logits.shape[0], len(inputs))
        loss = train_epoch(model, [data[0]])
        self.assertGreaterEqual(loss, 0.0)


if __name__ == "__main__":
    unittest.main()
