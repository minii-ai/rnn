import numpy as np


class RNN:
    """Vanilla character-level RNN in numpy"""

    @staticmethod
    def load(path: str):
        """Load weights from pickle file"""
        pass

    def __init__(self, hidden_size: int, vocab: str):
        """
        Initialize the weight and bias matrices

        Params:
            - hidden_size: hidden state dim
            - vocab: string of unique chars
        """
        vocab_size = len(vocab)

        self.hidden_size = hidden_size
        self.char_to_idx = {char: i for i, char in enumerate(vocab)}
        self.idx_to_char = {i: char for i, char in enumerate(vocab)}
        self.vocab_size = vocab_size

        # weights
        self.Wxh = np.random.randn(hidden_size, vocab_size)
        self.Whh = np.random.randn(hidden_size, hidden_size)
        self.Why = np.random.randn(vocab_size, hidden_size)

        # biases
        self.bh = np.random.randn(1, hidden_size)
        self.by = np.random.randn(1, vocab_size)

    def __call__(self, h, x) -> np.ndarray:
        pass

    def sample(self, char: str, n: int):
        """Generates samples starting with `char` for `n` iterations"""
        assert len(char) == 1 and char in self.char_to_idx
        x = np.zeros((1, self.hidden_size))
        x[:, self.char_to_idx[char]] = 1  # create one hot encoding for char
        h = np.zeros((1, self.hidden_size))  # initialize hidden state to all 0s

    def loss(self, inputs: list[int], targets: list[int]):
        """
        Computes loss between
        """
        pass

    def save(self, path: str):
        """Save the weights and vocab as a pickle file"""
        pass
