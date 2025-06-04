"""A small demonstration language model following the cell-based architecture.

This script builds on cells.py to create a minimal pipeline with an encoder,
a cognitive core composed of Cell objects, and a tiny API layer for
interaction. The goal is to keep the example lightweight so it can be run on
low-power hardware (approx. 20 W class devices).
"""

from cells import CellModel

class TokenEncoder:
    def __init__(self, vocab):
        self.vocab = vocab
        self.token_to_id = {ch: i for i, ch in enumerate(vocab)}
        self.vocab_size = len(vocab)

    def encode(self, ch):
        return [1.0 if i == self.token_to_id[ch] else 0.0
                for i in range(self.vocab_size)]

    def decode(self, vec):
        idx = vec.index(max(vec))
        return self.vocab[idx]

class CognitiveCore:
    def __init__(self, vocab_size, hidden_size=32):
        self.model = CellModel([vocab_size, hidden_size, vocab_size])

    def train(self, samples, epochs=10, lr=0.05):
        self.model.train(samples, epochs=epochs, lr=lr)

    def forward(self, x):
        return self.model.forward(x)

class LanguageBrain:
    def __init__(self, vocab):
        self.encoder = TokenEncoder(vocab)
        self.core = CognitiveCore(len(vocab))

    def train_identity(self, epochs=10, lr=0.05):
        samples = []
        for ch in self.encoder.vocab:
            vec = self.encoder.encode(ch)
            samples.append((vec, vec))
        self.core.train(samples, epochs=epochs, lr=lr)

    def reply(self, ch):
        vec = self.encoder.encode(ch)
        out = self.core.forward(vec)
        return self.encoder.decode(out)

if __name__ == "__main__":
    vocab = list("abcdefghijklmnopqrstuvwxyz ")
    brain = LanguageBrain(vocab)
    brain.train_identity(epochs=15, lr=0.1)

    # simple test
    test_ch = 'h'
    prediction = brain.reply(test_ch)
    print(f"Input {test_ch} -> predicted {prediction}")
