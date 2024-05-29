import numpy as np
import pickle


def softmax(z, t: float = 1.0):
    """Softmax w/ temperature t"""
    z_exp = np.exp(z / t)
    return z_exp / np.sum(z_exp)


class RNN:
    """Vanilla character-level RNN in numpy"""

    @staticmethod
    def load(path: str):
        """Load weights from pickle file"""
        with open(path, "rb") as f:
            data = pickle.load(f)  # read pickle file
            rnn = RNN(data["hidden_size"], data["vocab"])
            rnn.Wxh = data["Wxh"]  # load weights
            rnn.Whh = data["Whh"]
            rnn.Why = data["Why"]
            rnn.bh = data["bh"]
            rnn.by = data["by"]
            return rnn

    def __init__(self, hidden_size: int, vocab: str):
        """
        Initialize the weight and bias matrices

        Params:
            - hidden_size: hidden state dim
            - vocab: string of unique chars
        """
        self.hidden_size = hidden_size
        self.vocab = vocab
        self.vocab_size = len(vocab) + 1
        self.char_to_idx = {char: i for i, char in enumerate(vocab)}
        self.idx_to_char = {i: char for i, char in enumerate(vocab)}

        # add EOS token
        self.eos_token = "<EOS>"
        self.eos_token_idx = self.vocab_size - 1
        self.char_to_idx[self.eos_token] = self.eos_token_idx
        self.idx_to_char[self.eos_token_idx] = self.eos_token

        # weights
        self.Wxh = np.random.randn(hidden_size, self.vocab_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(self.vocab_size, hidden_size) * 0.01

        # biases
        self.bh = np.zeros((1, hidden_size))
        self.by = np.zeros((1, self.vocab_size))

        self.weights = (self.Wxh, self.Whh, self.Why, self.bh, self.by)
        self.num_params = sum(w.size for w in self.weights)

    def encode(self, chars: str):
        """Turns a string of chars into idxes"""
        ids = [self.char_to_idx[char] for char in chars] + [self.eos_token_idx]
        return ids

    def decode(self, idxes: list[int]):
        """Turns a list of idxes into chars"""
        chars = [self.idx_to_char[idx] for idx in idxes if idx != self.eos_token_idx]
        return "".join(chars)

    def __call__(self, x, h, t=1.0) -> np.ndarray:
        """RNN forward pass at softmax temperature t"""
        assert x.shape == (1, self.vocab_size) and h.shape == (1, self.hidden_size)
        zh = x @ self.Wxh.T + h @ self.Whh.T + self.bh
        hnext = np.tanh(zh)
        zy = hnext @ self.Why.T + self.by
        y = softmax(zy, t)

        return y, hnext

    def sample(self, char: str, n: int, t: float = 1.0):
        """Generates samples starting with char for n iterations at temperature t"""
        sample = ""
        for char in self.sample_progressive(char, n, t):
            sample += char

        return sample

    def sample_progressive(self, c: str, n: int, t: float = 1.0):
        """Generate one char at a time, starting with c for n iterations at temperature t"""
        assert len(c) == 1 and c in self.char_to_idx
        x = np.zeros((1, self.vocab_size))
        x[0, self.char_to_idx[c]] = 1  # create one hot encoding for char
        h = np.zeros((1, self.hidden_size))  # initialize hidden state to all 0s

        yield c

        for _ in range(n):
            probs, h = self(x, h, t)
            idx = np.random.choice(
                self.vocab_size, p=probs.ravel()
            )  # sample token idx from output

            x = np.zeros((1, self.vocab_size))
            x[0, idx] = 1  # one hot encoding for sampled token
            char = self.idx_to_char[idx]

            yield char

    def loss(self, inputs: list[int], targets: list[int], hprev=None):
        """
        Computes loss between input and target chars idxes and returns
        the loss, gradients (dWxh, dWhh, dWhy, dbh, dby), and final hidden state
        """
        assert len(inputs) == len(targets)
        xs, hs, ps = {}, {}, {}  # keep track of x, hidden states, and output probs
        hs[-1] = (
            hprev if hprev is not None else np.zeros((1, self.hidden_size))
        )  # store initial hidden state
        loss = 0

        # forward pass
        for t in range(len(inputs)):
            x = np.zeros((1, self.vocab_size))
            x[0, inputs[t]] = 1  # one hot encoding
            p, h = self(x, hs[t - 1])  # rnn

            xs[t] = x  # store x, hidden state, probs (we'll need them for backprop)
            hs[t] = h
            ps[t] = p
            loss += -np.log(ps[t][0, targets[t]])

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
            dzy[:, targets[t]] -= 1

            # 2nd layer
            dWhy += dzy.T @ hs[t]
            dby += dzy

            # intermediate gradients
            dLdh = dzy @ self.Why  # gradient of loss at L_t w.r.t hidden state h_t
            dhdzh = 1 - hs[t] ** 2  # gradient thr. tanh activation
            dFprevdh = dLdh + dFdh  # gradient of F_{t-1} w.r.t h_t
            dFprevdzh = dFprevdh * dhdzh  # gradient of F_{t-1} w.r.t z_h

            # 1st layer
            dWxh += dFprevdzh.T @ xs[t]
            dWhh += dFprevdzh.T @ hs[t - 1]
            dbh += dFprevdzh

            dFdh = dFprevdzh @ self.Whh  # update dFdh for next timestep t-1

        gradients = (dWxh, dWhh, dWhy, dbh, dby)  # collect gradients
        hnext = hs[len(inputs) - 1]  # final hidden state

        return loss, gradients, hnext

    def save(self, path: str):
        """Save the weights and vocab as a pickle file"""
        data = {
            "hidden_size": self.hidden_size,
            "vocab": self.vocab,
            "Wxh": self.Wxh,
            "Whh": self.Whh,
            "Why": self.Why,
            "bh": self.bh,
            "by": self.by,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)
