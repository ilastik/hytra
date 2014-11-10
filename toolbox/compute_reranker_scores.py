#!/usr/bin/env python
import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute Re-ranker scores given '
        'feature vectors and the learned weights')

    # file paths
    parser.add_argument('--features', required=True, type=str, dest='features', 
        help='Feature vectors of the proposals, all in one file')
    parser.add_argument('--weights', required=True, type=str, dest='weights',
        help='Learned re-ranker weights')
    parser.add_argument('--output', required=True, type=str, dest='output',
        help='Output file that will contain scores in the same order as feature vectors')

    options = parser.parse_args()

    feature_vectors = np.loadtxt(options.features)
    weights = np.loadtxt(options.weights)
    scores = np.dot(weights, feature_vectors)

    np.savetxt(options.output, scores)
    print("Scores for the {} proposals are: {}".format(scores.shape[0], scores))
