#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import argparse

import sys
sys.path.append('../toolbox/.')
# need this for weighted loss loading!
import structsvm


def scatterplot(proposals, name, color, plt, ground_truth_label, ground_truth_label_class_multiplier_list, loss_weights):
    if len(proposals) > 0:
        # compute losses vs gt:
        X = []
        Y = []

        for i in range(len(proposals)):
            gt_loss = structsvm.utils.multiclass_weighted_hamming_loss(
                proposals[i], ground_truth_label, ground_truth_label_class_multiplier_list, loss_weights)
            for j in range(len(proposals)):
                if j == i:
                    continue

                intra_loss = structsvm.utils.multiclass_weighted_hamming_loss(
                    proposals[i], proposals[j], ground_truth_label_class_multiplier_list, loss_weights)

                Y.append(intra_loss)
                X.append(gt_loss)

        plt.scatter(X, Y, c=color, label=name)

        max_val = int(max(max(X), max(Y)))
        return max_val
    else:
        print("No proposals found for " + name)
        return 0

def load_proposals(input_filenames):
    if input_filenames:
        return [structsvm.utils.load_proposal_labeling(p) for p in input_filenames]
    else:
        return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Scatter-Plot the loss between proposals and vs the loss w.r.t. ground truth')

    # file paths
    parser.add_argument('--divmbest-proposals', type=str, nargs='+', dest='divmbest_proposals',
                        help='Proposal labeling files')
    parser.add_argument('--gpc-proposals', type=str, nargs='+', dest='gpc_proposals',
                        help='Proposal labeling files')
    parser.add_argument('--perturbnmap-proposals', type=str, nargs='+', dest='perturbnmap_proposals',
                        help='Proposal labeling files')
    parser.add_argument('--ground-truth', required=True, type=str, dest='ground_truth',
                        help='Ground truth labeling file')
    parser.add_argument('--loss-weights', required=True, type=float, nargs='+', dest='loss_weights',
                        help='Loss weight vector indicating the loss of each class')
    parser.add_argument('-o', required=True, type=str, dest='out_file',
                        help='Name of the file the plot is saved to')

    options = parser.parse_args()

    # get labelings
    ground_truth_label, ground_truth_label_class_multiplier_list, _ = \
        structsvm.utils.load_labelfile_with_classes_and_multipliers(options.ground_truth)

    divmbest_proposals = load_proposals(options.divmbest_proposals)
    gpc_proposals = load_proposals(options.gpc_proposals)
    perturbnmap_proposals = load_proposals(options.perturbnmap_proposals)

    # plot
    plt.figure()
    plt.hold(True)
    m1 = scatterplot(divmbest_proposals, 'DiverseMBest', 'b', plt, ground_truth_label, ground_truth_label_class_multiplier_list, options.loss_weights)
    m2 = scatterplot(gpc_proposals, 'Gaussian Process Classifier', 'g', plt, ground_truth_label, ground_truth_label_class_multiplier_list, options.loss_weights)
    m3 = scatterplot(perturbnmap_proposals, 'Perturb-and-MAP', 'c', plt, ground_truth_label, ground_truth_label_class_multiplier_list, options.loss_weights)
    max_val = max([m1, m2, m3])
    plt.plot(range(max_val), range(max_val), 'r--', label='_')
    plt.xlabel("Loss w.r.t. ground truth")
    plt.ylabel("Loss w.r.t. other proposals")
    plt.legend()

    plt.savefig(options.out_file)