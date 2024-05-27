import argparse
import numpy as np
from rnn import RNN
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", type=str, default="./data/howtogetrich.txt")
    parser.add_argument("--hidden_size", "-hs", type=int, default=64)
    parser.add_argument("--iters", "-i", type=int, default=1000)
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


def clip_gradients(gradients):
    for gradient in gradients:
        np.clip(gradient, -1, 1, out=gradient)  # clip gradient in-place


def step(weights: np.ndarray, gradients: np.ndarray, lr: float):
    for weight, gradient in zip(weights, gradients):
        weight -= lr * gradient  # go in the opposite direction of the gradient


def train_loop(
    rnn: RNN,
    data: str,
    iters: int,
    lr: float,
    seq_length: int,
):
    print("[INFO] Training...")
    print(f"[INFO] hidden_size = {rnn.hidden_size}")
    print(f"[INFO] vocab_size = {rnn.vocab_size}")
    print(f"[INFO] lr = {lr}")
    print(f"[INFO] iters = {iters}")
    print(f"[INFO] seq_length = {seq_length}")
    print(f"[INFO] data_size = {len(data)}")

    i = 0
    h = np.zeros((1, rnn.hidden_size))
    with tqdm(total=iters, position=0) as pbar:
        for iter in range(iters):
            if i >= len(data) - 1:
                i = 0  # reset to start of data
                h = np.zeros((1, rnn.hidden_size))  # reset hidden state

            batch = data[i : i + seq_length + 1]
            inputs, targets = batch[:-1], batch[1:]  # prepare input, target for loss

            # compute loss and gradients
            loss, gradients, h = rnn.loss(inputs, targets, h)
            clip_gradients(gradients)  # gradient clipping for training stability
            step(rnn.weights, gradients, lr)  # gradient descent

            if (iter + 1) % 100 == 0:
                tqdm.write("== Checkpoint ==")
                tqdm.write(f"[Iter {iter}] loss = {loss}")
                sample = rnn.sample("h", 10)
                tqdm.write(sample)
                tqdm.write("\n")

            pbar.set_postfix(loss=loss)
            pbar.update(1)


def main(args):
    np.random.seed(0)
    # data = read_file(args.data)  # read data from txt file
    data = "hello world"
    vocab = build_vocab(data)  # build vocab of unique chars
    rnn = RNN(hidden_size=args.hidden_size, vocab=vocab)  # init rnn

    train_loop(
        rnn=rnn, data=data, iters=args.iters, lr=args.lr, seq_length=args.seq_length
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
