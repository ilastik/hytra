#!/usr/bin/env python

import os.path
import sys
sys.path.append('.')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../hytra/.')

import numpy as np
import matplotlib.pyplot as plt
import argparse
# need this for weighted loss loading!

import structsvm


def load_proposals(input_filenames):
    if input_filenames:
        return [structsvm.utils.load_proposal_labeling(p) for p in input_filenames]
    else:
        return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Scatter-Plot the loss between proposals and vs the loss w.r.t. ground truth')

    # file paths
    parser.add_argument('--proposals', type=str, nargs='+', dest='proposals',
                        help='Proposal labeling files')
    parser.add_argument('--ground-truth', required=True, type=str, dest='ground_truth',
                        help='Ground truth labeling file')
    parser.add_argument('--features', type=str, dest='features',
                        help='file containing the features for the proposals')
    parser.add_argument('--reranker-weights', type=str, dest='reranker_weights',
                        help='file containing the learned reranker weights')
    parser.add_argument('--loss-weights', required=True, type=float, nargs='+', dest='loss_weights',
                        help='Loss weight vector indicating the loss of each class')
    parser.add_argument('-o', required=True, type=str, dest='out_file',
                        help='Name of the file the plot is saved to')

    options = parser.parse_args()

    # get labelings
    ground_truth_label, ground_truth_label_class_multiplier_list, _ = \
        structsvm.utils.load_labelfile_with_classes_and_multipliers(options.ground_truth)

    proposals = load_proposals(options.proposals)
    losses = []
    for i in range(len(proposals)):
        gt_loss = structsvm.utils.multiclass_weighted_hamming_loss(
            proposals[i], ground_truth_label, ground_truth_label_class_multiplier_list, options.loss_weights)
        losses.append(gt_loss)

    # compute scores
    feature_vectors = np.loadtxt(options.features)
    weights = np.loadtxt(options.reranker_weights)
    means = np.loadtxt(os.path.splitext(options.reranker_weights)[0] + '_means.txt')
    variances = np.loadtxt(os.path.splitext(options.reranker_weights)[0] + '_variances.txt')
    structsvm.utils.apply_feature_normalization(feature_vectors, means, variances)
    scores = np.dot(weights, feature_vectors)

    # print scores
    # print losses
    
    # plot
    plt.figure()
    plt.hold(True)
    plt.scatter(losses, scores)
    plt.xlabel("Loss")
    plt.ylabel("Score")
    
    plt.savefig(options.out_file)