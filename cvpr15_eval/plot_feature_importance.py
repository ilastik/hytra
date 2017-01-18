#!/usr/bin/env python
from __future__ import print_function
from __future__ import unicode_literals
import os.path
import sys

sys.path.append('.')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../hytra/.')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_area_auto_adjustable
import argparse
import trackingfeatures

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot feature importance')

    # file paths
    parser.add_argument('--weights', required=True, type=str,nargs='+', dest='weights',
                        help='File containing the learned re-ranker weights')
    parser.add_argument('--legend', type=str, dest='legend',
                        help='File containing the legend of weight files',default="")
    parser.add_argument('--feature-names', type=str, dest='feature_names', default='',
                        help='Description for each feature')
    parser.add_argument('--expansion-range', type=int, nargs='+', dest='expansion_range',
                        help='Lower and upper bound of track weight expansions (default 0, 1)')
    parser.add_argument('-o', required=True, type=str, dest='out_file',
                        help='Name of the file the plot is saved to')
    parser.add_argument('--non-zero', action='store_true', dest='non_zero', help='display only non zero feature importance')

    parser.add_argument('--log', action='store_true', dest='logscale', help='use log scale on y axis')
    parser.add_argument('--sort', action='store_true', dest='sort', help='sort features')
    parser.add_argument('--limit', type=float, dest='limit',
                        help='threshold for featureweight if non zero is used',default=0)
    parser.add_argument('--fontsize', type=int,  dest='fontsize', help='fontsize of ticks',default=4)

    options = parser.parse_args()

    colors = ['b', 'g', 'r', 'c', 'm', 'y','dodgerblue','orangered','cyan']

    weightList = []
    NumberOfWeightFiles = 0
    for weightFile in options.weights:
        w = np.loadtxt(weightFile)
        length = np.sqrt(np.dot(w,w))
        w /= length
        weightList.append(w)
        NumberOfWeightFiles += 1

    assert(NumberOfWeightFiles <= len(colors))

    feature_names = []
    if len(options.feature_names) > 0:
        with open(options.feature_names, 'r') as fn:
            for line in fn:
                feature_names.append(line.strip())
    else:
        if len(options.expansion_range) == 2:
            assert(options.expansion_range[0] < options.expansion_range[1])
            feature_names = trackingfeatures.LineagePart.get_expanded_feature_names(options.expansion_range)
        else:
            if len(options.expansion_range) > 0:
                print("Invalid range specified, using default features...")
            feature_names = trackingfeatures.LineagePart.all_feature_names

    if len(options.legend) > 0:
        with open(options.legend) as f:
            labels = f.read().splitlines()


    plt.figure()
    ax = plt.axes()

    weights = zip(*weightList)

    if options.non_zero:
        weights,feature_names = zip(*[d for d in zip(weights,feature_names) if np.any(np.array(d[0])>options.limit) ])

    if options.sort:
        sortingmeasure = [np.sum(np.array(d)) for d in weights]
        weights,feature_names = zip(*[(x,f) for (s,x,f) in sorted(zip(sortingmeasure,weights,feature_names), key=lambda pair: pair[0],reverse=True)])

    width = 1./(len(weightList)+1)
    offset = -0.5 * width * (NumberOfWeightFiles-1)
    x_pos = 1.0 * np.arange(len(feature_names))
    l = ''
    for i in xrange(NumberOfWeightFiles):
        if len(options.legend) > 0:
            l = labels[i]
        plt.bar(x_pos+offset, [w[i] for w in weights], color=colors[i],align='center', width=width,log=options.logscale,linewidth=0,label=l)
        offset += width

    plt.xticks(x_pos, feature_names, rotation='vertical', fontsize=options.fontsize)
    plt.legend(prop={'size':6})
    make_axes_area_auto_adjustable(ax)
    plt.savefig(options.out_file)
