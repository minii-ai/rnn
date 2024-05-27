import argparse
import numpy as np
from rnn import RNN
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", type=str, default="./data/howtogetrich.txt")
    parser.add_argument("--hidden_size", "-hs", type=int, default=64)
    parser.add_argument("--iters", "-i", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seq_length", "-s", type=int, default=25)
    parser.add_argument("--save_path", "-sp", default="./weights.pickle")

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


def step(weights: np.ndarray, gradients: np.ndarray, lr: float):
    pass


def train_loop(
    rnn: RNN,
    data: str,
    iters: int,
    lr: float,
    seq_length: int,
):
    i = 0
    with tqdm(total=iters) as pbar:
        for _ in range(iters):
            if i >= len(data) - 1:
                i = 0  # reset to start of data

            batch = data[i : i + seq_length + 1]
            inputs, targets = batch[:-1], batch[1:]

            rnn.loss(inputs, targets)

            break
            pbar.update(1)


def main(args):
    data = read_file(args.data)  # read data from txt file
    vocab = build_vocab(data)  # build vocab of unique chars
    rnn = RNN(hidden_size=args.hidden_size, vocab=vocab)  # init rnn

    train_loop(
        rnn=rnn, data=data, iters=args.iters, lr=args.lr, seq_length=args.seq_length
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
