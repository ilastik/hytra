# pythonpath modification to make hytra available 
# for import without requiring it to be installed
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
# standard imports
import logging
import configargparse as argparse
from hytra.core.jsongraph import JsonTrackingGraph, writeToFormattedJSON


logger = logging.getLogger('convexify_costs.py')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='(Strictly!) Convexify the costs of a model to allow a flow-based solution',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', is_config_file=True, help='config file path', dest='config_file')
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
    logger.debug("Ignoring unknown parameters: {}".format(unknown))

    trackingGraph = JsonTrackingGraph(model_filename=args.model_filename)
    trackingGraph.convexifyCosts(args.epsilon)

    if args.result_filename is None:
        args.result_filename = args.model_filename

    writeToFormattedJSON(args.result_filename, trackingGraph.model)
