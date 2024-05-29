import numpy as np
from rnn import RNN
import argparse
import re
import random
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", type=str, default="./data/stevejobs_short.txt")
    parser.add_argument("--hidden_size", "-hs", type=int, default=64)
    parser.add_argument("--iters", "-i", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--seq_length", "-s", type=int, default=25)
    parser.add_argument("--save_path", "-sp")
    parser.add_argument("--val_n", "-vn", type=int, default=50)
    parser.add_argument("--val_steps", "-vs", type=int, default=100)
    parser.add_argument("--val_c", "-vc", type=str, default="h")
    parser.add_argument("--val_t", "-vt", type=float, default=0.5)

    return parser.parse_args()


def read_file(path: str):
    with open(path, "r") as f:
        return f.read()


def build_vocab(data: str) -> str:
    chars = set()
    vocab = ""
    for char in data:
        if char not in chars:
            vocab += char
            chars.add(char)

    return vocab


def clip_gradients(gradients):
    for gradient in gradients:
        np.clip(gradient, -1, 1, out=gradient)  # clip gradient in-place


data = read_file("./data/stevejobs_short.txt")
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print("data has %d characters, %d unique.", (data_size, vocab_size))


# hyperparameters
hidden_size = 100  # size of hidden layer of neurons


def train_loop(
    rnn: RNN,
    dataset: list[str],
    iters: int,
    lr: float,
    seq_length: int,
):
    # memory variables for Adagrad
    mWxh, mWhh, mWhy = (
        np.zeros_like(rnn.Wxh),
        np.zeros_like(rnn.Whh),
        np.zeros_like(rnn.Why),
    )
    mbh, mby = np.zeros_like(rnn.bh), np.zeros_like(rnn.by)

    n = 0
    smooth_loss = -np.log(1.0 / vocab_size) * seq_length  # loss at iteration 0

    while True:
        batches = random.sample(dataset, len(dataset))
        for minibatch in batches:
            hprev = np.zeros((1, hidden_size))  # reset RNN memory
            minibatch_idxs = rnn.encode(minibatch)

            for p in range(0, len(minibatch), seq_length):
                batch = minibatch_idxs[p : p + seq_length + 1]
                inputs, targets = batch[:-1], batch[1:]

                # sample from the model now and then
                if n % 1000 == 0:
                    txt = rnn.sample('"', 600, 0.5)
                    print("----\n %s \n----" % (txt,))

                # forward seq_length characters through the net and fetch gradient
                loss, gradients, hprev = rnn.loss(inputs, targets, hprev)
                smooth_loss = smooth_loss * 0.999 + loss * 0.001
                clip_gradients(gradients)

                if n % 1000 == 0:
                    print("iter %d, loss: %f" % (n, smooth_loss))  # print progress

                for param, dparam, mem in zip(
                    rnn.weights,
                    gradients,
                    [mWxh, mWhh, mWhy, mbh, mby],
                ):
                    mem += dparam**2
                    param += -lr * dparam / np.sqrt(mem + 1e-8)  # adagrad update

                n += 1  # iteration counter


def main(args):
    data = read_file(args.data)  # read txt file
    dataset = re.split(r"\n\s*\n", data)  # split data into paragraphs
    vocab = build_vocab(data)  # build vocab of unique chars
    rnn = RNN(hidden_size, vocab)

    train_loop(
        rnn=rnn,
        dataset=dataset,
        iters=args.iters,
        lr=args.lr,
        seq_length=args.seq_length,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
