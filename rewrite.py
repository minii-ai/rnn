import numpy as np
from rnn import RNN
import argparse
import re
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", type=str, default="./data/stevejobs.txt")
    parser.add_argument("--hidden_size", "-hs", type=int, default=64)
    parser.add_argument("--iters", "-i", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--seq_length", "-s", type=int, default=25)
    parser.add_argument("--save_path", "-sp", required=True)
    parser.add_argument("--val_n", "-vn", type=int, default=50)
    parser.add_argument("--val_steps", "-vs", type=int, default=100)
    parser.add_argument("--val_c", "-vc", type=str, default="h")
    parser.add_argument("--val_t", "-vt", type=float, default=0.5)

    return parser.parse_args()


def read_file(path: str):
    with open(path, "r") as f:
        return f.read()


def build_dataset(path: str):
    data = read_file(path)
    dataset = re.split(r"\n\s*\n", data)  # split data into paragraphs
    return dataset


def clip_gradients(gradients):
    for gradient in gradients:
        np.clip(gradient, -1, 1, out=gradient)  # clip gradient in-place


data = read_file("./data/stevejobs_short.txt")
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print("data has %d characters, %d unique.", (data_size, vocab_size))


dataset = build_dataset("./data/stevejobs_short.txt")


# hyperparameters
hidden_size = 100  # size of hidden layer of neurons
seq_length = 25  # number of steps to unroll the RNN for
lr = 1e-1


def train_loop():
    pass


def main():

    rnn = RNN(hidden_size, chars)

    # memory variables for Adagrad
    mWxh, mWhh, mWhy = (
        np.zeros_like(rnn.Wxh),
        np.zeros_like(rnn.Whh),
        np.zeros_like(rnn.Why),
    )
    mbh, mby = np.zeros_like(rnn.bh), np.zeros_like(
        rnn.by
    )  # memory variables for Adagrad

    n, p = 0, 0
    smooth_loss = -np.log(1.0 / vocab_size) * seq_length  # loss at iteration 0

    print(len(dataset))

    while True:
        shuffled_list = random.sample(dataset, len(dataset))

        for minibatch in shuffled_list:

            # prepare inputs (we're sweeping from left to right in steps seq_length long)
            hprev = np.zeros((1, hidden_size))  # reset RNN memory

            for p in range(0, len(minibatch), seq_length):
                batch = data[p : p + seq_length + 1]
                inputs, targets = (
                    batch[:-1],
                    batch[1:],
                )  # prepare input, target for loss
                inputs = rnn.encode(inputs)
                targets = rnn.encode(targets)

                # sample from the model now and then
                if n % 1000 == 0:
                    txt = rnn.sample('"', 600, 1.0)
                    print("----\n %s \n----" % (txt,))

                # forward seq_length characters through the net and fetch gradient
                loss, gradients, hprev = rnn.loss(inputs, targets, hprev)
                smooth_loss = smooth_loss * 0.999 + loss * 0.001
                if n % 1000 == 0:
                    print("iter %d, loss: %f" % (n, smooth_loss))  # print progress

                clip_gradients(gradients)

                for param, dparam, mem in zip(
                    rnn.weights,
                    gradients,
                    [mWxh, mWhh, mWhy, mbh, mby],
                ):
                    mem += dparam**2
                    param += -lr * dparam / np.sqrt(mem + 1e-8)  # adagrad update

                p += seq_length  # move data pointer
                n += 1  # iteration counter


if __name__ == "__main__":
    main()
