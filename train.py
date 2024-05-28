import argparse
import numpy as np
from rnn import RNN
from tqdm import tqdm


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


def train_loop(
    rnn: RNN,
    data: str,
    iters: int,
    lr: float,
    seq_length: int,
    val_n: int,
    val_steps: int,
    val_c: str,
    val_t: float,
):
    print("[INFO] Training...")
    print(f"[INFO] data_size = {len(data)}")
    print(f"[INFO] num_params = {rnn.num_params}")
    print(f"[INFO] vocab_size = {rnn.vocab_size}")
    print(f"[INFO] hidden_size = {rnn.hidden_size}")
    print(f"[INFO] lr = {lr}")
    print(f"[INFO] iters = {iters}")
    print(f"[INFO] seq_length = {seq_length}")

    i = 0
    h = np.zeros((1, rnn.hidden_size))

    # for adagrad (https://gist.github.com/karpathy/d4dee566867f8291f086)
    mWxh, mWhh, mWhy = (
        np.zeros_like(rnn.Wxh),
        np.zeros_like(rnn.Whh),
        np.zeros_like(rnn.Why),
    )
    mbh, mby = np.zeros_like(rnn.bh), np.zeros_like(rnn.by)
    smooth_loss = -np.log(1.0 / rnn.vocab_size) * seq_length

    with tqdm(total=iters, position=0) as pbar:
        for iter in range(iters):
            if i >= len(data) - 1:
                i = 0  # reset to start of data
                h = np.zeros((1, rnn.hidden_size))  # reset hidden state

            batch = data[i : i + seq_length]
            inputs, targets = batch[:-1], batch[1:]  # prepare input, target for loss
            inputs, targets = rnn.encode(inputs), rnn.encode(targets)

            # compute loss and gradients
            loss, gradients, h = rnn.loss(inputs, targets, h)
            smooth_loss = smooth_loss * 0.999 + loss * 0.001
            clip_gradients(gradients)  # gradient clipping for training stability

            # adagrad gradient descent
            for weight, gradient, mem in zip(
                rnn.weights, gradients, [mWxh, mWhh, mWhy, mbh, mby]
            ):
                mem += gradient**2
                weight -= lr * gradient / np.sqrt(mem + 1e-8)

            if (
                (iter + 1) % val_steps == 0 or iter == iters - 1 or iter == 0
            ):  # validation step
                tqdm.write(f"== Iter {iter + 1} ==")
                tqdm.write(f"loss = {smooth_loss}")
                sample = rnn.sample(val_c, val_n, val_t)
                tqdm.write(sample)

            i += seq_length  # move to next batch

            pbar.set_postfix(loss=loss)
            pbar.update(1)

    print(f"[INFO] Final Loss = {loss}")
    print("[INFO] Sample")
    print(rnn.sample(val_c, val_n))
    print("[INFO] Training complete!")


def main(args):
    np.random.seed(0)
    data = read_file(args.data)  # read data from txt file
    vocab = build_vocab(data)  # build vocab of unique chars
    rnn = RNN(hidden_size=args.hidden_size, vocab=vocab)  # init rnn

    train_loop(
        rnn=rnn,
        data=data,
        iters=args.iters,
        lr=args.lr,
        seq_length=args.seq_length,
        val_n=args.val_n,
        val_steps=args.val_steps,
        val_c=args.val_c,
        val_t=args.val_t,
    )

    print("[INFO] Saving model...")
    rnn.save(args.save_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
