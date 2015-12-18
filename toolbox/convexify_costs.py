import sys
import commentjson as json
import os
import argparse
import numpy as np

sys.path.append('../.')
sys.path.append('.')
from progressbar import ProgressBar

def listify(l):
    return [[e] for e in l]

def convexify(l, eps):
    features = np.array(l)
    if features.shape[1] != 1:
        raise InvalidArgumentException('This script can only convexify feature vectors with one feature per state!')

    # Note from Numpy Docs: In case of multiple occurrences of the minimum values, the indices corresponding to the first occurrence are returned.
    bestState = np.argmin(features)

    for direction in [-1, 1]:
        pos = bestState + direction
        previousGradient = 0
        while pos >= 0 and pos < features.shape[0]:
            newGradient = features[pos] - features[pos-direction]
            if newGradient == previousGradient:
                # cost function's derivative is constant, add epsilon
                previousGradient += eps
                features[pos] = features[pos-direction] + previousGradient
            elif newGradient < previousGradient:
                # cost function got too flat, set feature value to match old slope
                features[pos] = features[pos-direction] + previousGradient
            else:
                # all good, continue with new slope
                previousGradient = newGradient
                
            pos += direction
    return listify(features.flatten())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='(Strictly!) Convexify the costs of a model to allow a flow-based solution')
    parser.add_argument('--model', required=True, type=str, dest='model_filename',
                        help='Filename of the json model description')
    parser.add_argument('--output', required=True, type=str, dest='result_filename',
                        help='Filename of the json file containing the model with convexified costs')
    parser.add_argument('--epsilon', type=float, dest='epsilon', default=0.000001,
                        help='Epsilon is added to the gradient if the 1st derivative has a plateau.')
    
    args = parser.parse_args()

    print("Loading model file: " + args.model_filename)
    with open(args.model_filename, 'r') as f:
        model = json.load(f)

    if not model['settings']['statesShareWeights']:
        raise InvalidArgumentException('This script can only convexify feature vectors with shared weights!')

    progressBar = ProgressBar(stop=(len(model['segmentationHypotheses']) + len(model['linkingHypotheses'])))
    segmentationHypotheses = model['segmentationHypotheses']
    for seg in segmentationHypotheses:
        for f in ['features', 'appearanceFeatures', 'disappearanceFeatures']:
            if f in seg:
                seg[f] = convexify(seg[f], args.epsilon)
        # division features are always convex (is just a line)
        progressBar.show()

    linkingHypotheses = model['linkingHypotheses']
    for link in linkingHypotheses:
        link['features'] = convexify(link['features'], args.epsilon)
        progressBar.show()

    with open(args.result_filename, 'w') as f:
        json.dump(model, f, indent=4, separators=(',', ': '))