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
        self.vocab_size = len(vocab)
        # self.vocab_size = len(vocab) + 1
        self.char_to_idx = {char: i for i, char in enumerate(vocab)}
        self.idx_to_char = {i: char for i, char in enumerate(vocab)}

        # add EOS token
        # self.eos_token = "<EOS>"
        # self.eos_token_idx = self.vocab_size - 1
        # self.char_to_idx[self.eos_token] = self.eos_token_idx
        # self.idx_to_char[self.eos_token_idx] = self.eos_token

        # weights
        self.Wxh = np.random.randn(hidden_size, self.vocab_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(self.vocab_size, hidden_size) * 0.01

        # biases
        self.bh = np.random.randn(hidden_size, 1)
        self.by = np.random.randn(self.vocab_size, 1)

        self.weights = (self.Wxh, self.Whh, self.Why, self.bh, self.by)
        self.num_params = sum(w.size for w in self.weights)

    def encode(self, chars: str):
        """Turns a string of chars into idxes"""
        ids = [self.char_to_idx[char] for char in chars]
        # ids = [self.char_to_idx[char] for char in chars] + [self.eos_token_idx]
        return ids

    def decode(self, idxes: list[int]):
        """Turns a list of idxes into chars"""
        chars = [self.idx_to_char[idx] for idx in idxes if idx != self.eos_token_idx]
        return "".join(chars)

    def __call__(self, x, h, t=1.0) -> np.ndarray:
        """RNN forward pass at softmax temperature t"""
        assert x.shape == (self.vocab_size, 1) and h.shape == (self.hidden_size, 1)
        zh = self.Wxh @ x + self.Whh @ h + self.bh
        hnext = np.tanh(zh)
        zy = self.Why @ hnext + self.by
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
        x = np.zeros((self.vocab_size, 1))
        x[self.char_to_idx[c]] = 1  # create one hot encoding for char
        h = np.zeros((self.hidden_size, 1))  # initialize hidden state to all 0s

        yield c

        for _ in range(n):
            probs, h = self(x, h, t)
            idx = np.random.choice(
                self.vocab_size, p=probs.ravel()
            )  # sample token idx from output

            x = np.zeros((self.vocab_size, 1))
            x[idx] = 1  # one hot encoding for sampled token
            char = self.idx_to_char[idx]

            yield char

    def loss(self, inputs: list[int], targets: list[int], hprev=None):
        """
        Computes loss between input and target chars idxes and returns
        the loss, gradients (dWxh, dWhh, dWhy, dbh, dby), and final hidden state
        """
        assert len(inputs) == len(targets)
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(hprev)
        loss = 0
        # forward pass
        for t in range(len(inputs)):
            xs[t] = np.zeros((self.vocab_size, 1))  # encode in 1-of-k representation
            xs[t][inputs[t]] = 1
            hs[t] = np.tanh(
                np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t - 1]) + self.bh
            )  # hidden state
            ys[t] = (
                np.dot(self.Why, hs[t]) + self.by
            )  # unnormalized log probabilities for next chars
            ps[t] = np.exp(ys[t]) / np.sum(
                np.exp(ys[t])
            )  # probabilities for next chars
            loss += -np.log(ps[t][targets[t], 0])  # softmax (cross-entropy loss)
        # backward pass: compute gradients going backwards
        dWxh, dWhh, dWhy = (
            np.zeros_like(self.Wxh),
            np.zeros_like(self.Whh),
            np.zeros_like(self.Why),
        )
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dhnext = np.zeros_like(hs[0])
        for t in reversed(range(len(inputs))):
            dy = np.copy(ps[t])
            dy[
                targets[t]
            ] -= 1  # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
            dWhy += np.dot(dy, hs[t].T)
            dby += dy
            dh = np.dot(self.Why.T, dy) + dhnext  # backprop into h
            dhraw = (1 - hs[t] * hs[t]) * dh  # backprop through tanh nonlinearity
            dbh += dhraw
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t - 1].T)
            dhnext = np.dot(self.Whh.T, dhraw)

        gradients = (dWxh, dWhh, dWhy, dbh, dby)  # collect gradients
        hnext = hs[len(inputs) - 1]  # final hidden state

        return loss, gradients, hnext

    # def loss(self, inputs: list[int], targets: list[int], hprev=None):
    #     """
    #     Computes loss between input and target chars idxes and returns
    #     the loss, gradients (dWxh, dWhh, dWhy, dbh, dby), and final hidden state
    #     """
    #     assert len(inputs) == len(targets)
    #     xs, hs, ys, ps = {}, {}, {}, {}

    #     hs[-1] = np.copy(hprev)
    #     loss = 0
    #     # forward pass
    #     for t in range(len(inputs)):
    #         xs[t] = np.zeros((1, self.vocab_size))  # encode in 1-of-k representation
    #         xs[t][0, inputs[t]] = 1
    #         hs[t] = np.tanh(
    #             np.dot(xs[t], self.Wxh.T)
    #             + np.dot(
    #                 hs[t - 1],
    #                 self.Whh.T,
    #             )
    #             + self.bh
    #         )  # hidden state
    #         ys[t] = (
    #             np.dot(hs[t], self.Why.T) + self.by
    #         )  # unnormalized log probabilities for next chars
    #         ps[t] = np.exp(ys[t]) / np.sum(
    #             np.exp(ys[t])
    #         )  # probabilities for next chars
    #         loss += -np.log(ps[t][0, targets[t]])  # softmax (cross-entropy loss)

    #     # gradient of loss w.r.t weights and biases
    #     dWxh, dWhh, dWhy = (
    #         np.zeros_like(self.Wxh),
    #         np.zeros_like(self.Whh),
    #         np.zeros_like(self.Why),
    #     )
    #     dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
    #     dhnext = np.zeros_like(hs[0])
    #     for t in reversed(range(len(inputs))):
    #         dy = np.copy(ps[t])  # (1, l)
    #         dy[
    #             0, targets[t]
    #         ] -= 1  # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
    #         dWhy += np.dot(dy.T, hs[t])  # (l, n) (l, 1) (1, h_T)
    #         dby += dy

    #         dh = np.dot(dy, self.Why) + dhnext  # (1, n) (n, 1) ()

    #         # (n, 1) (n, l) (l, 1)
    #         dhraw = (1 - hs[t] * hs[t]) * dh  # backprop through tanh nonlinearity
    #         dbh += dhraw
    #         dWxh += np.dot(dhraw.T, xs[t])
    #         dWhh += np.dot(dhraw.T, hs[t - 1])
    #         dhnext = np.dot(dhraw, self.Whh)
    #         # (n, 1) (n, n) (n, 1)

    #         gradients = (dWxh, dWhh, dWhy, dbh, dby)  # collect gradients
    #         hnext = hs[len(inputs) - 1]  # final hidden state

    #     return loss, gradients, hnext

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
