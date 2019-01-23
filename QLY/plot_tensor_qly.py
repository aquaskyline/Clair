import sys
import os
import argparse
import math
import numpy as np
import param
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import utils_qly as utils # v3 network is using v2 utils
import qianliyan_v2 as cv

def Prepare(args):
    utils.setup_environment()
    m = cv.Qianliyan()
    m.init()

    m.restore_parameters(args.chkpnt_fn)


    return m, utils, 1


def GetActivations(layer, batchX, m):
    # Version 3 network
    units = m.session.run(layer, feed_dict={m.XPH:batchX,
                                            m.phasePH:False,
                                            m.dropoutRateFC4PH:0.0,
                                            m.dropoutRateFC5PH:0.0,
                                            m.l2RegularizationLambdaPH:0.0})
    return units


def PlotTensor(ofn, XArray):
    plot = plt.figure(figsize=(15, 8))

    plt.subplot(4,1,1)
    plt.xticks(np.arange(0, 33, 1))
    plt.yticks(np.arange(0, 8, 1), ['A','C','G','T','a','c','g','t'])
    plt.imshow(XArray[0,:,:,0].transpose(), vmin=0, vmax=50, interpolation="nearest", cmap=plt.cm.hot)
    plt.colorbar()

    plt.subplot(4,1,2)
    plt.xticks(np.arange(0, 33, 1))
    plt.yticks(np.arange(0, 8, 1), ['A','C','G','T','a','c','g','t'])
    plt.imshow(XArray[0,:,:,1].transpose(), vmin=-50, vmax=50, interpolation="nearest", cmap=plt.cm.bwr)
    plt.colorbar()

    plt.subplot(4,1,3)
    plt.xticks(np.arange(0, 33, 1))
    plt.yticks(np.arange(0, 8, 1), ['A','C','G','T','a','c','g','t'])
    plt.imshow(XArray[0,:,:,2].transpose(), vmin=-50, vmax=50, interpolation="nearest", cmap=plt.cm.bwr)
    plt.colorbar()

    plt.subplot(4,1,4)
    plt.xticks(np.arange(0, 33, 1))
    plt.yticks(np.arange(0, 8, 1), ['A','C','G','T','a','c','g','t'])
    plt.imshow(XArray[0,:,:,3].transpose(), vmin=-50, vmax=50, interpolation="nearest", cmap=plt.cm.bwr)
    plt.colorbar()

    plot.savefig(ofn, dpi=300, transparent=True, bbox_inches='tight')
    plt.close(plot)


def CreatePNGs(args, m, utils):
    f = open(args.array_fn, 'r')
    array = f.read()
    f.close()
    import re
    array = re.split("\n",array)
    array = [x for x in array if x]
    print(array)

    splitted_array = []
    for i in range(len(array)):
        splitted_array += re.split(",", array[i])

    print("splitted array length")
    print(len(splitted_array))
    print(splitted_array[0])
    # for i in range(len(splitted_array)):
    #     splitted_array[i] = int(splitted_array[i])

    XArray = np.array(splitted_array).reshape((-1,33,8,4))
    YArray = np.zeros((1,16))
    varName = args.name
    print >> sys.stderr, "Plotting %s..." % (varName)

    # Create folder
    if not os.path.exists(varName):
        os.makedirs(varName)

    # Plot tensors
    PlotTensor(varName+"/tensor.png", XArray)

def ParseArgs():
    parser = argparse.ArgumentParser(
            description="Visualize tensors and hidden layers in PNG" )

    parser.add_argument('--array_fn', type=str, default = "vartensors",
            help="Array input")

    parser.add_argument('--name', type=str, default = None,
            help="output name")

    parser.add_argument('--chkpnt_fn', type=str, default = None,
            help="Input a checkpoint for testing or continue training")

    parser.add_argument('--slim', type=param.str2bool, nargs='?', const=True, default = False,
            help="Train using the slim version of Clairvoyante, default: False")

    args = parser.parse_args()

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)

    return args


def main():
    args = ParseArgs()
    m, utils, total = Prepare(args)
    CreatePNGs(args, m, utils)


if __name__ == "__main__":
    main()
