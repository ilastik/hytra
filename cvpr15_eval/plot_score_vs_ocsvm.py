#!/usr/bin/env python

import os.path
import sys
sys.path.append('.')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../toolbox/.')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import math
import structsvm
import trackingfeatures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Show quality of reranker compared to one class SVM')

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

    filename, extension = os.path.splitext(options.out_file)

    prec_score_pairs = zip(list(precisions), scores)
    prec_score_pairs.sort(key=lambda x: x[1], reverse=True) # sort by score
    sorted_precs, sorted_scores = zip(*prec_score_pairs)
    
    threshold = 0.9
    def filter_precs(threshold, precs):
        precs = list(precs)
        i = 0
        num_below_thresh = sum(np.array(precs) < threshold)
        for c in range(len(precs)):
            if precs[c] < threshold:
                i+=1
            precs[c] = 1.0 - float(i) / len(precs)
        return precs

    sorted_precs = filter_precs(threshold, sorted_precs)

    # plot only outlier svm score (averaged over lineage) vs precision
    outlier_svm_scores = []
    track_outlier_feature_idx = trackingfeatures.LineagePart.feature_to_weight_idx('track_outlier_svm_score')
    div_outlier_feature_idx = trackingfeatures.LineagePart.feature_to_weight_idx('div_outlier_svm_score')
    for lt in lineage_trees:
        fv = lt.get_feature_vector()
        outlier_svm_scores.append(fv[track_outlier_feature_idx] + fv[div_outlier_feature_idx])

    prec_outlier_pairs = zip(list(precisions), outlier_svm_scores)
    prec_outlier_pairs.sort(key=lambda x: x[1], reverse=True) # sort by outlier_svm_score
    o_sorted_precs, sorted_outliers = zip(*prec_outlier_pairs)
    o_sorted_precs = filter_precs(threshold, o_sorted_precs)

    plt.figure()
    plt.hold(True)
    plt.plot(sorted_precs,label='ordered by score')
    plt.plot(o_sorted_precs,label='ordered by outlier-svm')
    plt.xlabel("ordered paths")
    plt.ylabel("Precision")
    #plt.savefig(filename + "_outlier_score" + extension)
    plt.legend()
    plt.savefig(options.out_file)
    print("Saved figure ", options.out_file) 

    # # scatter plot
    # plt.figure()
    # plt.hold(True)
    # plt.scatter(precisions, scores)
    # plt.xlabel("Precision")
    # plt.ylabel("Score")
    # plt.savefig(options.out_file)

    # # length histogram
    # plt.figure()
    # plt.hist(lengths, 100)
    # plt.xlabel("Length")
    # plt.ylabel("Frequency")
    # plt.savefig(filename + "_length_histo" + extension)

    # # sort according to precision and plot again
    # # log_scores = map(math.log, scores)
    # prec_score_pairs = zip(list(precisions), scores, num_divs, num_tracks, lengths)
    # prec_score_pairs.sort(key=lambda x: x[1]) # sort by score

    # plt.figure()
    # plt.plot(range(len(prec_score_pairs)), zip(*prec_score_pairs)[0])
    # plt.ylabel("Precision")
    # plt.xlabel("Num Tracks, sorted by score")
    # plt.savefig(filename + "_sorted_num_tracks" + extension)

    # plt.figure()
    # plt.hold(True)
    # plt.plot(zip(*prec_score_pairs)[1], zip(*prec_score_pairs)[0])
    # plt.ylabel("Precision")
    # plt.xlabel("Score")
    # plt.savefig(filename + "_sorted" + extension)

    # plt.figure()
    # plt.hold(True)
    # plt.scatter(zip(*prec_score_pairs)[2], zip(*prec_score_pairs)[1], c='b', label='Num divisions')
    # plt.scatter(zip(*prec_score_pairs)[3], zip(*prec_score_pairs)[1], c='r', label='Num tracks')
    # # plt.plot(zip(*prec_score_pairs)[4], zip(*prec_score_pairs)[1], c='g', label='overall lengths')
    # plt.xlabel("Length")
    # plt.ylabel("Score")
    # plt.legend()
    # plt.savefig(filename + "_length_score" + extension)

    # plt.figure()
    # plt.hold(True)
    # plt.scatter(zip(*prec_score_pairs)[2], zip(*prec_score_pairs)[0], c='b', label='Num divisions')
    # plt.scatter(zip(*prec_score_pairs)[3], zip(*prec_score_pairs)[0], c='r', label='Num tracks')
    # # plt.scatter(zip(*prec_score_pairs)[4], zip(*prec_score_pairs)[0], c='g', label='overall lengths')
    # plt.xlabel("Length")
    # plt.ylabel("Precision")
    # plt.legend()
    # plt.savefig(filename + "_length_precision" + extension)
