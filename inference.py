import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", "-w", type=str, required=True)
    parser.add_argument("--char", "-c", type=str, required=True)
    parser.add_argument("--length", "-n", type=int, default=10)

    return parser.parse_args()


def main(args):
    print(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
