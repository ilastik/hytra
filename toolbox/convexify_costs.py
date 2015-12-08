import sys
import commentjson as json
import os
import argparse
import numpy as np

def listify(l):
    return [[e] for e in l]

def convexify(l):
	features = np.array(l)
	if features.shape[1] != 1:
		raise InvalidArgumentException('This script can only convexify feature vectors with one feature per state!')

	bestState = np.argmin(features)

	for direction in [-1, 1]:
		pos = bestState + direction
		previousGradient = 0
		while pos >= 0 and pos < features.shape[0]:
			newGradient = features[pos] - features[pos-direction]
			if abs(newGradient) < abs(previousGradient):
				# cost function got too flat, set feature value to match old slope
				features[pos] = features[pos-direction] + previousGradient
			else:
				# all good, continue with new slope
				previousGradient = newGradient
				
			pos += direction
	return listify(features.flatten())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Take a json file containing a result to a set of HDF5 events files')
    parser.add_argument('--model', required=True, type=str, dest='model_filename',
                        help='Filename of the json model description')
    parser.add_argument('--output', required=True, type=str, dest='result_filename',
                        help='Filename of the json file containing the model with convexified costs')
    
    args = parser.parse_args()

    with open(args.model_filename, 'r') as f:
        model = json.load(f)

    if not model['settings']['statesShareWeights']:
    	raise InvalidArgumentException('This script can only convexify feature vectors with shared weights!')

    segmentationHypotheses = model['segmentationHypotheses']
    for seg in segmentationHypotheses:
    	for f in ['features', 'appearanceFeatures', 'disappearanceFeatures']:
    		if f in seg:
    			seg[f] = convexify(seg[f])
    	# division features are always convex (is just a line)

    linkingHypotheses = model['linkingHypotheses']
    for link in linkingHypotheses:
		link['features'] = convexify(link['features'])

    with open(args.result_filename, 'w') as f:
    	json.dump(model, f, indent=4, separators=(',', ': '))