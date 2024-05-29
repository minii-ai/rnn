import argparse
from rnn import RNN
import time
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", "-w", type=str, required=True)
    parser.add_argument("--num_samples", "-n", type=int, default=10)
    parser.add_argument("--temperature", "-t", type=float, default=0.5)

    return parser.parse_args()


MAX_TOKENS = 1000000


def main(args):
    rnn = RNN.load(args.weights)

    for _ in range(args.num_samples):
        for j in rnn.sample_progressive(args.temperature, n=MAX_TOKENS):
            char = rnn.decode([j])
            sys.stdout.write(char)
            sys.stdout.flush()

        print("\n")


if __name__ == "__main__":
    args = parse_args()
    main(args)
