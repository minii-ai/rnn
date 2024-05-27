import numpy as np


def softmax(z):
    e_z = np.exp(z)
    return e_z / e_z.sum(axis=1)


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

        self.weights = (self.Wxh, self.Whh, self.Why, self.bh, self.by)

    def __call__(self, x, h) -> np.ndarray:
        """Sample from RNN(x_t, h_{t-1}) -> y_t, h_t"""
        assert x.shape == (1, self.vocab_size) and h.shape == (1, self.hidden_size)
        zh = x @ self.Wxh.T + h @ self.Whh.T + self.bh
        hnext = np.tanh(zh)
        zy = hnext @ self.Why.T + self.by
        y = softmax(zy)

        return y, hnext

    def sample(self, char: str, n: int):
        """Generates samples starting with `char` for `n` iterations"""
        assert len(char) == 1 and char in self.char_to_idx
        x = np.zeros((1, self.vocab_size))
        x[:, self.char_to_idx[char]] = 1  # create one hot encoding for char
        h = np.zeros((1, self.hidden_size))  # initialize hidden state to all 0s
        idxes = []

        for _ in range(n):
            probs, h = self(x, h)
            idx = np.random.choice(
                self.vocab_size, p=probs.ravel()
            )  # sample token idx from output

            x = np.zeros((1, self.vocab_size))
            x[:, idx] = 1  # one hot encoding for sampled token
            idxes.append(idx)

        sample = "".join([self.idx_to_char[idx] for idx in idxes])
        return char + sample

    def loss(self, inputs: str, targets: str, hprev=None):
        """
        Computes loss between input and target chars and returns
        the loss, gradients (dWxh, dWhh, dWhy, dbh, dby), and final hidden state
        """
        hprev = np.zeros((1, self.hidden_size)) if hprev is not None else np.copy(hprev)
        xs, hs, ps = {}, {}, {}  # keep track of x, hidden states, and output probs
        hs[-1] = hprev  # store initial hidden state
        loss = 0

        # compute loss at each timestep
        for t in range(len(inputs)):
            x = np.zeros((1, self.vocab_size))
            x[:, self.char_to_idx[inputs[t]]] = 1  # one hot encoding of char
            probs, h = self(x, hs[t - 1])  # x, hprev -> rnn -> probs, hnext

            xs[t] = x  # store xs, hs, and probs, we'll use them during backprop
            hs[t] = h
            ps[t] = probs

            # cross entropy loss, nll of the predicted prob for the target char
            loss += -np.log(probs[0, self.char_to_idx[targets[t]]])

        # gradient of loss w.r.t weights and biases
        dWxh, dWhh, dWhy = (
            np.zeros_like(self.Wxh),
            np.zeros_like(self.Whh),
            np.zeros_like(self.Why),
        )
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)

        # gradient of F_t (future loss L_t+1 ... L_T) w.r.t. h_t
        dFdh = np.zeros((1, self.hidden_size))

        # backprop thr. time
        for t in reversed(range(len(inputs))):
            dzy = np.copy(ps[t])  # loss at t w.r.t zy
            dzy[:, self.char_to_idx[targets[t]]] -= 1

            # 2nd layer
            dWhy += hs[t] * dzy.T
            dby += dzy

            # intermediate gradients
            dLdh = dzy @ self.Why  # gradient of loss at L_t w.r.t hidden state h_t
            dhdzh = 1 - hs[t] ** 2  # gradient thr. tanh activation
            dhdhprev = dhdzh * self.Whh  # gradient of hidden state h_t w.r.t h_{t-1}
            dFprevdh = dLdh + dFdh  # gradient of F_{t-1} w.r.t h_t

            # 1st layer
            dWxh += xs[t] * dhdzh.T * dFprevdh.T
            dWhh += hs[t - 1] * dhdzh.T * dFprevdh.T
            dbh += dhdzh * dFprevdh

            # update dFdh for next timestep t-1
            dFdh = dFprevdh @ dhdhprev

        gradients = (dWxh, dWhh, dWhy, dbh, dby)  # collect gradients
        hnext = hs[len(inputs) - 1]

        return loss, gradients, hnext

    def save(self, path: str):
        """Save the weights and vocab as a pickle file"""
        pass
