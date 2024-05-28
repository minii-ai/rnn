"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""

from rnn_copy import RNN

import numpy as np

# data I/O
data = open(
    "./data/stevejobs_short.txt", "r"
).read()  # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print("data has %d characters, %d unique.", (data_size, vocab_size))

# hyperparameters
hidden_size = 100  # size of hidden layer of neurons
seq_length = 25  # number of steps to unroll the RNN for
learning_rate = 1e-1


rnn = RNN(hidden_size, chars)

print(rnn.char_to_idx)

n, p = 0, 0
mWxh, mWhh, mWhy = (
    np.zeros_like(rnn.Wxh),
    np.zeros_like(rnn.Whh),
    np.zeros_like(rnn.Why),
)  # memory variables for Adagrad
mbh, mby = np.zeros_like(rnn.bh), np.zeros_like(rnn.by)
smooth_loss = -np.log(1.0 / vocab_size) * seq_length  # loss at iteration 0
while True:
    # prepare inputs (we're sweeping from left to right in steps seq_length long)
    if p + seq_length + 1 >= len(data) or n == 0:
        # hprev = np.zeros((1, hidden_size))  # reset RNN memory
        hprev = np.zeros((hidden_size, 1))  # reset RNN memory
        p = 0  # go from start of data
    inputs = [rnn.char_to_idx[ch] for ch in data[p : p + seq_length]]
    targets = [rnn.char_to_idx[ch] for ch in data[p + 1 : p + seq_length + 1]]

    # sample from the model now and then
    if n % 1000 == 0:
        # sample_ix = sample(hprev, char_to_ix['"'], 600)
        print(rnn.char_to_idx['"'])
        # sample_ix = rnn.sample(hprev, '"', 600)
        sample_ix = rnn.sample('"', 600)
        # print(sample_ix)
        # txt = "".join(rnn.idx_to_char[ix] for ix in sample_ix)
        print("----\n %s \n----" % (sample_ix,))

    # forward seq_length characters through the net and fetch gradient
    loss, gradients, hprev = rnn.loss(inputs, targets, hprev)

    for dparam in gradients:
        np.clip(dparam, -5, 5, out=dparam)

    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    if n % 1000 == 0:
        print("iter %d, loss: %f" % (n, smooth_loss))  # print progress

    # perform parameter update with Adagrad
    for param, dparam, mem in zip(
        rnn.weights,
        gradients,
        [mWxh, mWhh, mWhy, mbh, mby],
    ):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8)  # adagrad update

    p += seq_length  # move data pointer
    n += 1  # iteration counter
