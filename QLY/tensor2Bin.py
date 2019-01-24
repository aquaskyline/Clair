import sys
import argparse
import logging
import pickle
import param
import utils
logging.basicConfig(format='%(message)s', level=logging.INFO)

def Run(args):
    utils.SetupEnv()
    Convert(args, utils)


def Convert(args, utils):
    logging.info("Loading the dataset ...")
    total, XArrayCompressed, YArrayCompressed, posArrayCompressed = \
    utils.GetTrainingArray(args.tensor_fn,
                           args.var_fn,
                           args.bed_fn)

    logging.info("Writing to binary ...")
    fh = open(args.bin_fn, "wb")
    pickle.dump(total,fh)
    pickle.dump(XArrayCompressed,fh)
    pickle.dump(YArrayCompressed,fh)
    pickle.dump(posArrayCompressed,fh)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description="Generate a binary format input tensor" )

    parser.add_argument('--tensor_fn', type=str, default = "vartensors",
            help="Tensor input")

    parser.add_argument('--var_fn', type=str, default = "truthvars",
            help="Truth variants list input")

    parser.add_argument('--bed_fn', type=str, default = None,
            help="High confident genome regions input in the BED format")

    parser.add_argument('--bin_fn', type=str, default = None,
            help="Output a binary tensor file")

    args = parser.parse_args()

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)

    Run(args)
