#!/usr/bin/env python
__author__ = 'chaubold'

import os.path
import sys
sys.path.append('.')
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../hytra/.')

import numpy as np
import matplotlib.pyplot as plt
import argparse
import trackingfeatures

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Get a feature matrix where each lineage tree is a sample')

    # file paths
    parser.add_argument('--lineage', type=str, required=True, dest='lineage_filename',
                        help='Lineage tree dump')
    parser.add_argument('-o', required=True, type=str, dest='out_file', default='track_feature_vectors.txt',
                        help='Name of the file the feature matrix is saved to')

    options = parser.parse_args()

    # load
    tracks, divisions, lineage_trees = trackingfeatures.load_lineage_dump(options.lineage_filename)
    print("Found {} tracks, {} divisions and {} lineage trees".format(len(tracks),
                                                                      len(divisions),
                                                                      len(lineage_trees)))

    # compute feature vectors
    feature_vectors = np.zeros((len(lineage_trees), len(lineage_trees[0].get_expanded_feature_vector([-1, 2]))))
    for i, lt in enumerate(lineage_trees):
        feat_vec = lt.get_expanded_feature_vector([-1, 2])
        feature_vectors[i, :] = feat_vec

    np.savetxt(options.out_file, feature_vectors.transpose())
