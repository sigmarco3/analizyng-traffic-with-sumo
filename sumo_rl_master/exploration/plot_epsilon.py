import argparse
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':

    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # prs.add_argument("-e", dest="epsilon", type=float, required=True, help="Epsilon\n")
    # prs.add_argument("-d", dest="decay", type=float, required=True, help="Epsilon\n")
    args = prs.parse_args()

    plt.plot([i for i in range(0, 1000000, 5)], [0.05 * 0.99995**i for i in range(0, 200000)])

    plt.grid()
    plt.show()
