import numpy as np
import pickle


def softmax(z):
    e_z = np.exp(z)
    return e_z / e_z.sum(axis=1)


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
        self.char_to_idx = {char: i for i, char in enumerate(vocab)}
        self.idx_to_char = {i: char for i, char in enumerate(vocab)}

        # weights
        self.Wxh = np.random.randn(hidden_size, self.vocab_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(self.vocab_size, hidden_size) * 0.01

        # biases
        self.bh = np.random.randn(1, hidden_size)
        self.by = np.random.randn(1, self.vocab_size)

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
        sample = ""
        for char in self.sample_progressive(char, n):
            sample += char

        return sample

    def sample_progressive(self, c: str, n: int):
        """Generate one char at a time, starting with `c` for `n` iterations"""
        assert len(c) == 1 and c in self.char_to_idx
        x = np.zeros((1, self.vocab_size))
        x[:, self.char_to_idx[c]] = 1  # create one hot encoding for char
        h = np.zeros((1, self.hidden_size))  # initialize hidden state to all 0s

        yield c

        for _ in range(n):
            probs, h = self(x, h)
            idx = np.random.choice(
                self.vocab_size, p=probs.ravel()
            )  # sample token idx from output

            x = np.zeros((1, self.vocab_size))
            x[:, idx] = 1  # one hot encoding for sampled token
            char = self.idx_to_char[idx]

            yield char

    def loss(self, inputs: str, targets: str, hprev):
        """
        Computes loss between input and target chars and returns
        the loss, gradients (dWxh, dWhh, dWhy, dbh, dby), and final hidden state
        """
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(hprev)
        loss = 0
        inputs = [self.char_to_idx[char] for char in inputs]
        targets = [self.char_to_idx[char] for char in targets]
        # forward pass
        for t in range(len(inputs)):
            xs[t] = np.zeros((1, self.vocab_size))  # encode in 1-of-k representation
            xs[t][0, inputs[t]] = 1
            hs[t] = np.tanh(
                np.dot(xs[t], self.Wxh.T) + np.dot(hs[t - 1], self.Whh.T) + self.bh
            )  # hidden state
            ys[t] = (
                np.dot(hs[t], self.Why.T) + self.by
            )  # unnormalized log probabilities for next chars
            ps[t] = np.exp(ys[t]) / np.sum(
                np.exp(ys[t])
            )  # probabilities for next chars
            loss += -np.log(ps[t][0, targets[t]])  # softmax (cross-entropy loss)

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
        dhnext = np.zeros_like(hs[0])
        for t in reversed(range(len(inputs))):
            dy = np.copy(ps[t])
            dy[
                0, targets[t]
            ] -= 1  # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
            dWhy += np.dot(dy.T, hs[t])
            dby += dy
            dh = np.dot(dy, self.Why) + dhnext  # backprop into h
            dhraw = (1 - hs[t] * hs[t]) * dh  # backprop through tanh nonlinearity
            dbh += dhraw
            dWxh += np.dot(dhraw.T, xs[t])
            dWhh += np.dot(dhraw.T, hs[t - 1])
            dhnext = np.dot(dhraw, self.Whh)

        # for t in reversed(range(len(inputs))):
        #     dzy = np.copy(ps[t])  # loss at t w.r.t zy
        #     dzy[0, targets[t]] -= 1

        #     # 2nd layer
        #     dWhy += np.dot(dzy.T, hs[t])
        #     dby += dzy

        #     # intermediate gradients
        #     dLdh = dzy @ self.Why  # gradient of loss at L_t w.r.t hidden state h_t
        #     dFprevdh = dLdh + dFdh  # gradient of F_{t-1} w.r.t h_t
        #     dhraw = (1 - hs[t] ** 2) * dFprevdh  # gradient thr. tanh activation
        #     #     dhdzh = 1 - hs[t] ** 2  # gradient thr. tanh activation
        #     #     dhdhprev = dhdzh * self.Whh  # gradient of hidden state h_t w.r.t h_{t-1}

        #     #     # 1st layer
        #     dbh += dhraw
        #     dWxh += np.dot(dhraw.T, xs[t])
        #     dWhh += np.dot(dhraw.T, hs[t - 1])
        #     dFdh = np.dot(dhraw, self.Whh)  # update dFdh for next timestep t-1
        #     # dFdh = dFprevdh @ (1 - hs[t] ** 2).T * self.Whh
        #     a = dFprevdh
        #     b = (1 - hs[t] ** 2).T * self.Whh
        #     dFdh = a @ b
        # print(a @ b)

        # update dFdh for next timestep t-1
        # dFdh = dFprevdh @ dhdhprev

        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)

        gradients = (dWxh, dWhh, dWhy, dbh, dby)  # collect gradients
        hnext = hs[len(inputs) - 1]

        # print(dby)

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
