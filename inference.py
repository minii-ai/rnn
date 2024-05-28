import argparse
from rnn import RNN
import time
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", "-w", type=str, required=True)
    parser.add_argument("--char", "-c", type=str, required=True)
    parser.add_argument("--length", "-n", type=int, default=10)

    return parser.parse_args()


def main(args):
    rnn = RNN.load(args.weights)

    for char in rnn.sample_progressive(args.char, args.length - 1):
        sys.stdout.write(char)
        sys.stdout.flush()


if __name__ == "__main__":
    args = parse_args()
    main(args)
