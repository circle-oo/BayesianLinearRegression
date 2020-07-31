import blr
import data
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dimen")
    parser.add_argument("alpha")
    parser.add_argument("beta")
    parser.add_argument("mn")
    args = parser.parse_args()
    model = blr.BLR(float(args.alpha), float(args.beta), int(args.dimen))
    data = data.Data()
    for i in range(4):
        data.addData((i+1)*(i+1))
        model.fit(data.getData(), data.getTarget())
        model.getImage(data.getData(), data.getTarget(), int(args.mn))