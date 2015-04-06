#!/usr/bin/env python

import os.path
import sys
sys.path.append('.')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../toolbox/.')

import numpy as np
import matplotlib.pyplot as plt
import argparse

import structsvm
import trackingfeatures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Scatter-Plot the precision of a lineage vs the reranker score')

    # file paths
    parser.add_argument('--lineage', type=str, required=True, dest='lineage_filename',
                        help='Lineage tree dump')
    parser.add_argument('--precisions', type=str, required=True, dest='precisions_filename',
                        help='file containing the precision against the ground truth of each lineage tree')
    parser.add_argument('--reranker-weights', type=str, dest='reranker_weights',
                        help='file containing the learned reranker weights')
    parser.add_argument('-o', required=True, type=str, dest='out_file',
                        help='Name of the file the plot is saved to')

    options = parser.parse_args()

    # load
    precisions = np.loadtxt(options.precisions_filename)
    tracks, divisions, lineage_trees = trackingfeatures.load_lineage_dump(options.lineage_filename)
    print("Found {} tracks, {} divisions and {} lineage trees".format(len(tracks),
                                                                      len(divisions),
                                                                      len(lineage_trees)))
    weights = np.loadtxt(options.reranker_weights)
    means = np.loadtxt(os.path.splitext(options.reranker_weights)[0] + '_means.txt')
    variances = np.loadtxt(os.path.splitext(options.reranker_weights)[0] + '_variances.txt')

    # compute scores
    scores = []
    for lt in lineage_trees:
        feat_vec = np.expand_dims(lt.get_expanded_feature_vector([-1, 2]), axis=1)
        structsvm.utils.apply_feature_normalization(feat_vec, means, variances)
        score = np.dot(weights, feat_vec[:, 0])
        scores.append(score)

    # print scores
    # print losses

    # plot
    plt.figure()
    plt.hold(True)
    plt.scatter(precisions, scores)
    plt.xlabel("Precision")
    plt.ylabel("Score")

    plt.savefig(options.out_file)