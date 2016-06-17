# pythonpath modification to make hytra available 
# for import without requiring it to be installed
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
# standard imports
import logging
import numpy as np
import commentjson as json
import configargparse as argparse
from hytra.core.progressbar import ProgressBar
from hytra.core.jsongraph import convexify

def getLogger():
    return logging.getLogger('convexify_costs.py')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='(Strictly!) Convexify the costs of a model to allow a flow-based solution',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', is_config_file=True, help='config file path', dest='config_file', required=True)
    parser.add_argument('--graph-json-file', required=True, type=str, dest='model_filename',
                        help='Filename of the json model description')
    parser.add_argument('--out-json-file', default=None, type=str, dest='result_filename',
                        help='Filename of the json file containing the model with convexified costs.'
                        +' If None, it works in-place.')
    parser.add_argument('--epsilon', type=float, dest='epsilon', default=0.000001,
                        help='Epsilon is added to the gradient if the 1st derivative has a plateau.')
    parser.add_argument("--verbose", dest='verbose', action='store_true', default=False)

    # parse command line
    args, unknown = parser.parse_known_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    getLogger().debug("Ignoring unknown parameters: {}".format(unknown))

    getLogger().debug("Loading model file: " + args.model_filename)
    with open(args.model_filename, 'r') as f:
        model = json.load(f)

    if not model['settings']['statesShareWeights']:
        raise ValueError('This script can only convexify feature vectors with shared weights!')

    progressBar = ProgressBar(stop=(len(model['segmentationHypotheses']) + len(model['linkingHypotheses'])))
    segmentationHypotheses = model['segmentationHypotheses']
    for seg in segmentationHypotheses:
        for f in ['features', 'appearanceFeatures', 'disappearanceFeatures']:
            if f in seg:
                try:
                    seg[f] = convexify(seg[f], args.epsilon)
                except:
                    getLogger().warning("Convexification failed for feature {} of :{}".format(f, seg))
                    exit(0)
        # division features are always convex (is just a line)
        progressBar.show()

    linkingHypotheses = model['linkingHypotheses']
    for link in linkingHypotheses:
        link['features'] = convexify(link['features'], args.epsilon)
        progressBar.show()

    if args.result_filename is None:
        args.result_filename = args.model_filename

    with open(args.result_filename, 'w') as f:
        json.dump(model, f, indent=4, separators=(',', ': '))
