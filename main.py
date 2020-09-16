import argparse

from blr import BLR
from data import Data
from utils import init


def main(args: argparse.Namespace):
    model = BLR(args.alpha, args.beta, args.dim)
    dataset = Data()

    for i in range(args.data_size):
        dataset.add((i + 1) ** 2)
        model.fit(*dataset.get())
        model.plot(*dataset.get(), args.size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=6,
                        help="Bayesian Linear Regression model dimension")

    parser.add_argument("--alpha", type=float, default=10.,
                        help="Alpha param")
    parser.add_argument("--beta", type=float, default=10.,
                        help="Beta param")

    parser.add_argument("--data-size", type=int, default=4,
                        help="Data size")
    parser.add_argument("--size", type=int, default=5,
                        help="Plot figure size")

    parser.add_argument('-s', '--seed', required=False, default=42,
                        help="The answer to life the universe and everything")

    args = parser.parse_args()
    init(args.seed)
    main(args)
