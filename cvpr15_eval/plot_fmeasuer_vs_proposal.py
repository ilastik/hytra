#!/usr/bin/env python
import os.path
import sys

sys.path.append('.')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../toolbox/.')

import numpy as np
import os
import string

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_area_auto_adjustable
import argparse
import trackingfeatures

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot feature importance')

    # file paths
    parser.add_argument('--folders', required=True, type=str,nargs='+', dest='folders',
                        help='File containing the learned re-ranker weights')
    parser.add_argument('--legend', type=str,nargs='+', dest='legend',
                        help='File containing the legend of weight files',default=[])
    # parser.add_argument('--feature-names', type=str, dest='feature_names', default='',
    #                     help='Description for each feature')
    # parser.add_argument('--expansion-range', type=int, nargs='+', dest='expansion_range',
    #                     help='Lower and upper bound of track weight expansions (default 0, 1)')
    parser.add_argument('-o', required=True, type=str, dest='out_file',
                        help='Name of the file the plot is saved to')
    # parser.add_argument('--non-zero', action='store_true', dest='non_zero', help='display only non zero feature importance')

    # parser.add_argument('--log', action='store_true', dest='logscale', help='use log scale on y axis')
    # parser.add_argument('--sort', action='store_true', dest='sort', help='sort features')
    # parser.add_argument('--limit', type=float, dest='limit',
    #                     help='threshold for featureweight if non zero is used',default=0)
    parser.add_argument('--fontsize', type=int,  dest='fontsize', help='fontsize of ticks',default=15)
    parser.add_argument('--linewidth', type=int,  dest='linewidth', help='linewidth of graph lines',default=8)
    parser.add_argument('--latex', action='store_true', dest='latex', help='export graph in latex format')



    options = parser.parse_args()

    #Direct input 
    plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
    #Options
    params = {'text.usetex' : True,
              'font.size' : options.fontsize,
              'font.family' : 'lmodern',
              'text.latex.unicode': True,
              }
    plt.rcParams.update(params) 

    colors = ['b', 'g', 'r', 'c', 'm', 'y','dodgerblue','orangered','cyan']

    data = []
    for folder in options.folders:
        label = folder
        data.append([])

        for subfolder in os.listdir(folder+"/test/"):
            if subfolder.startswith("iter_"):
                # print folder+"/"+subfolder+"/result.txt"
                with open(str(folder)+"/test/"+str(subfolder)+"/result.txt") as f:
                    # print f.read().splitlines()
                    i = int(subfolder[5])
                    precission, recall, fmeasure =  string.split(f.read().splitlines()[0],",")[2:5]
                    data[-1].append((i, precission, recall, fmeasure))

        # data[-1] = sorted(zip(*data[-1]), key=lambda pair: pair[0])
        # print            data[-1]
        # print zip(*data[-1])     
        data[-1] = sorted(data[-1], key=lambda pair: pair[0])
        index, pre, rec, fme = zip(*data[-1]) 
        # plt.plot(index, pre, 'r-',label="pre")
        # plt.plot(index, rec, 'g-',label="rec")
        if len(options.legend) > 0:
            plotname = options.legend[len(data)-1]
        else:
            plotname = folder
        plt.plot(index, fme, color=colors[len(data)-1],label=plotname,linewidth=options.linewidth)
        # plt.axis([0, 6, 0, 20])

    plt.title('F-Measure of M-th proposal')
    plt.xlabel('Proposal')
    plt.ylabel('F-Measure')
    plt.tight_layout(.5)
    plt.legend()

    if options.latex:
        plt.savefig(options.out_file, 
            #This is simple recomendation for publication plots
            dpi=1000, 
            # Plot will be occupy a maximum of available space
            bbox_inches='tight', 
            )
    else:
        plt.savefig(options.out_file)
