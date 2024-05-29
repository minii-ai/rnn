import numpy as np
from rnn import RNN
import argparse
import re
import random
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", type=str, default="./data/stevejobs.txt")
    parser.add_argument("--hidden_size", "-hs", type=int, default=64)
    parser.add_argument("--iters", "-i", type=int, default=1000)
    parser.add_argument("--lr", "-lr", type=float, default=1e-1)
    parser.add_argument("--seq_length", "-s", type=int, default=25)
    parser.add_argument("--save_path", "-sp", required=True)
    parser.add_argument("--val_steps", "-vs", type=int, default=100)
    parser.add_argument("--val_t", "-vt", type=float, default=1.0)

    return parser.parse_args()


def read_file(path: str):
    with open(path, "r") as f:
        return f.read()


def build_vocab(data: str) -> str:
    chars = set()
    vocab = []
    for char in data:
        if char not in chars:
            vocab.append(char)
            chars.add(char)

    return vocab


def clip_gradients(gradients):
    for gradient in gradients:
        np.clip(gradient, -1, 1, out=gradient)  # clip gradient in-place


def train_loop(
    rnn: RNN,
    dataset: list[str],
    iters: int,
    lr: float,
    seq_length: int,
    val_steps: int,
    val_t: float,
    save_path: str,
):
    print("[INFO] Training...")
    print(f"[INFO] data_size = {len(dataset)}")
    print(f"[INFO] num_params = {rnn.num_params}")
    print(f"[INFO] vocab_size = {rnn.vocab_size}")
    print(f"[INFO] hidden_size = {rnn.hidden_size}")
    print(f"[INFO] lr = {lr}")
    print(f"[INFO] iters = {iters}")
    print(f"[INFO] seq_length = {seq_length}")

    # memory variables for Adagrad
    mWxh, mWhh, mWhy = (
        np.zeros_like(rnn.Wxh),
        np.zeros_like(rnn.Whh),
        np.zeros_like(rnn.Why),
    )
    mbh, mby = np.zeros_like(rnn.bh), np.zeros_like(rnn.by)

    n = 0
    pbar = tqdm(total=iters, position=0)
    smooth_loss = -np.log(1.0 / rnn.vocab_size) * seq_length  # loss at iteration 0
    smooth_perplexity = 2 ** smooth_loss

    while True:
        batches = random.sample(dataset, len(dataset))
        for minibatch in batches:
            hprev = np.zeros((1, rnn.hidden_size))  # reset RNN memory
            minibatch_idxs = rnn.encode(minibatch)

            for p in range(0, len(minibatch), seq_length):
                batch = minibatch_idxs[p : p + seq_length + 1]
                inputs, targets = batch[:-1], batch[1:]

                # compute loss and clip gradients
                loss, gradients, hprev = rnn.loss(inputs, targets, hprev)
                perplexity = 2 ** loss # 2 ^ cross entropy
                smooth_loss = smooth_loss * 0.999 + loss * 0.001
                smooth_perplexity = perplexity * 0.999 + loss * 0.001
                clip_gradients(gradients)

                # sample
                if (n + 1) % val_steps == 0 or n == 0 or n == iters - 1:
                    tqdm.write(f"iter: {n + 1}, loss: {smooth_loss}, perplexity: {smooth_perplexity}")
                    idxes = rnn.sample(val_t)
                    txt = rnn.decode(idxes)
                    tqdm.write(f"----\n{txt}\n----")
                    tqdm.write("")
                    rnn.save(save_path)

                # adagrad gradient descent
                for param, dparam, mem in zip(
                    rnn.weights,
                    gradients,
                    [mWxh, mWhh, mWhy, mbh, mby],
                ):
                    mem += dparam**2
                    param += -lr * dparam / np.sqrt(mem + 1e-8)  # adagrad update

                n += 1
                pbar.update(1)

                if n >= iters:
                    break
            if n >= iters:
                break
        if n >= iters:
            break

    print("[INFO] Training complete!")


def main(args):
    data = read_file(args.data)  # read txt file
    dataset = re.split(r"\n\s*\n", data)  # split data into paragraphs
    vocab = build_vocab(data)  # build vocab of unique chars

    rnn = RNN(args.hidden_size, vocab)

    train_loop(
        rnn=rnn,
        dataset=dataset,
        iters=args.iters,
        lr=args.lr,
        seq_length=args.seq_length,
        val_steps=args.val_steps,
        val_t=args.val_t,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
