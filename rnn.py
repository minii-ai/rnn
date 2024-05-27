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
        self.hidden_size = hidden_size
        self.char_to_idx = {char: i for i, char in enumerate(vocab)}
        self.idx_to_char = {i: char for i, char in enumerate(vocab)}

    def __call__(self, h, x) -> np.ndarray:
        pass

    def sample(self, char: str, n: int):
        """Generates samples starting with `char` for `n` iterations"""
        pass

    def loss(self, inputs: list[int], targets: list[int]):
        """
        Computes loss between
        """
        pass

    def save(self, path: str):
        """Save the weights and vocab as a pickle file"""
        pass
