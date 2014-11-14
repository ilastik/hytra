#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_area_auto_adjustable
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot feature importance')

    # file paths
    parser.add_argument('--weights', required=True, type=str, dest='weights',
                        help='File containing the learned re-ranker weights')
    parser.add_argument('--feature-names', required=True, type=str, dest='feature_names',
                        help='Description for each feature')
    parser.add_argument('-o', required=True, type=str, dest='out_file',
                        help='Name of the file the plot is saved to')

    options = parser.parse_args()

    weights = np.loadtxt(options.weights)
    feature_names = []
    with open(options.feature_names, 'r') as fn:
        for line in fn:
            feature_names.append(line.strip())

    plt.figure()
    x_pos = 2.0 * np.arange(len(feature_names))
    ax = plt.axes()
    plt.bar(x_pos, weights, align='center', width=0.8, alpha=0.4)
    plt.xticks(x_pos, feature_names, rotation='vertical', fontsize=2)
    make_axes_area_auto_adjustable(ax)
    plt.savefig(options.out_file)