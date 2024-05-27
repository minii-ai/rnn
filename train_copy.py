import argparse
import numpy as np
from rnn_copy import RNN


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", type=str, default="./data/howtogetrich.txt")
    parser.add_argument("--hidden_size", "-hs", type=int, default=64)
    parser.add_argument("--iters", "-i", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--seq_length", "-s", type=int, default=25)
    parser.add_argument("--save_path", "-sp", required=True)
    parser.add_argument("--val_n", "-vn", type=int, default=50)
    parser.add_argument("--val_steps", "-vs", type=int, default=100)
    parser.add_argument("--val_c", "-vc", type=str, default="h")

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
        weight += -lr * gradient  # go in the opposite direction of the gradient


def train_loop(
    rnn: RNN,
    data: str,
    iters: int,
    lr: float,
    seq_length: int,
    val_n: int,
    val_steps: int,
    val_c: str,
):
    print("[INFO] Training...")
    print(f"[INFO] hidden_size = {rnn.hidden_size}")
    print(f"[INFO] vocab_size = {rnn.vocab_size}")
    print(f"[INFO] lr = {lr}")
    print(f"[INFO] iters = {iters}")
    print(f"[INFO] seq_length = {seq_length}")
    print(f"[INFO] data_size = {len(data)}")

    n, p = 0, 0
    mWxh, mWhh, mWhy = (
        np.zeros_like(rnn.Wxh),
        np.zeros_like(rnn.Whh),
        np.zeros_like(rnn.Why),
    )
    mbh, mby = np.zeros_like(rnn.bh), np.zeros_like(rnn.by)
    smooth_loss = -np.log(1.0 / rnn.vocab_size) * seq_length  # loss at iteration 0
    while True:
        if p + seq_length + 1 >= len(data) or n == 0:
            hprev = np.zeros((1, rnn.hidden_size))  # reset RNN memory
            p = 0  # go from start of data
        inputs = data[p : p + seq_length]
        targets = data[p + 1 : p + seq_length + 1]

        # compute loss and gradients
        loss, gradients, hprev = rnn.loss(inputs, targets, hprev)
        # clip_gradients(gradients)  # gradient clipping for training stability
        # step(rnn.weights, gradients, lr)  # gradient descent

        for param, dparam, mem in zip(
            [rnn.Wxh, rnn.Whh, rnn.Why, rnn.bh, rnn.by],
            gradients,
            [mWxh, mWhh, mWhy, mbh, mby],
        ):
            mem += dparam * dparam
            # param += -lr * dparam  # adagrad update
            param += -lr * dparam / np.sqrt(mem + 1e-8)  # adagrad update

        smooth_loss = smooth_loss * 0.999 + loss * 0.001
        if n % 1000 == 0:
            print("---")
            print("iter %d, loss: %f" % (n, smooth_loss))  # print progress
            # sample = rnn.sample(val_c, val_n)
            sample = rnn.sample("I", 300, 0.2)
            print(sample)
            print("---")

        p += seq_length  # move data pointer
        n += 1  # iteration counter

        # if (iter + 1) % val_steps == 0:  # validation step
        #     tqdm.write(f"== Iter {iter} ==")
        #     tqdm.write(f"loss = {loss}")
        #     sample = rnn.sample(val_c, val_n)
        #     tqdm.write(sample)

        # i += seq_length  # move to next batch

        # pbar.set_postfix(loss=loss)
        # pbar.update(1)

    print(f"[INFO] Final Loss = {loss}")
    print("[INFO] Sample")
    print(rnn.sample(val_c, val_n))
    print("[INFO] Training complete!")


def main(args):
    np.random.seed(0)
    data = read_file(args.data)  # read data from txt file
    # data = data[:250]
    # data = data[:10000]
    # data = "hello world"
    # data = data[:50]
    # data = data[:25]
    vocab = build_vocab(data)  # build vocab of unique chars
    rnn = RNN(hidden_size=args.hidden_size, vocab=vocab)  # init rnn

    train_loop(
        rnn=rnn,
        data=data,
        iters=args.iters,
        lr=args.lr,
        seq_length=args.seq_length,
        # seq_length=10,
        val_n=args.val_n,
        val_steps=args.val_steps,
        val_c=args.val_c,
    )

    print("[INFO] Saving model...")
    rnn.save(args.save_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
