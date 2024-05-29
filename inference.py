import argparse
from rnn import RNN
import time
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", "-w", type=str, required=True)
    parser.add_argument("--char", "-c", type=str, required=True)
    parser.add_argument("--num_samples", "-n", type=int, default=10)
    parser.add_argument("--temperature", "-t", type=float, default=0.5)

    return parser.parse_args()


def main(args):
    rnn = RNN.load(args.weights)

    for _ in range(args.num_samples):
        i = rnn.encode(args.char)[0]
        for j in rnn.sample_progressive(i, args.temperature):
            char = rnn.decode([j])
            sys.stdout.write(char)
            sys.stdout.flush()

        print("\n")


if __name__ == "__main__":
    args = parse_args()
    main(args)
