import argparse
import numpy as np
from rnn import RNN
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", type=str, default="./data/howtogetrich.txt")
    parser.add_argument("--hidden_size", "-hs", type=int, default=64)
    parser.add_argument("--iters", "-i", type=int, default=100)
    parser.add_argument("--seq_length", "-s", type=int, default=25)

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


def train_loop():
    pass


def main(args):
    data = read_file(args.data)  # read data from txt file
    vocab = build_vocab(data)  # build vocab of unique chars
    rnn = RNN(hidden_size=args.hidden_size, vocab=vocab)  # init rnn

    with tqdm(total=args.iters) as pbar:
        pass


if __name__ == "__main__":
    args = parse_args()
    main(args)
