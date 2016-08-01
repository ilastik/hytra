# pythonpath modification to make hytra and empryonic available 
# for import without requiring it to be installed
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
# standard imports
import logging
import configargparse as argparse
from hytra.core.jsongraph import JsonTrackingGraph, writeToFormattedJSON
from hytra.core.jsongapcloser import JsonGapCloser

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Given a hypotheses json graph and a result.json, this script'
                        + ' resolves all mergers by updating the segmentation and inserting the appropriate '
                        + 'nodes and links.',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', is_config_file=True, help='config file path', dest='config_file')

    parser.add_argument('--ctc-input-file', required=True, type=str, dest='input_ctc',
                        help='IN Filename of the ctc result file: res_track.txt')
    parser.add_argument('--graph-json-file', required=True, type=str, dest='model_filename',
                        help='IN Filename of the json model description')
    parser.add_argument('--result-json-file', required=True, type=str, dest='result_filename',
                        help='IN Filename of the json file containing results')
    parser.add_argument('--label-image-file', required=True, type=str, dest='label_image_filename',
                        help='IN Filename of the original ilasitk tracking project')
    parser.add_argument('--label-image-path', dest='label_image_path', type=str,
                        default='/TrackingFeatureExtraction/LabelImage/0000/[[%d, 0, 0, 0, 0], [%d, %d, %d, %d, 1]]',
                        help='internal hdf5 path to label image')
    parser.add_argument('--raw-data-file', type=str, dest='raw_filename', default=None,
                      help='filename to the raw h5 file')
    parser.add_argument('--raw-data-path', type=str, dest='raw_path', default='volume/data',
                      help='Path inside the raw h5 file to the data')
    parser.add_argument("--raw-data-axes", dest='raw_axes', type=str, default='txyzc',
                        help="axes ordering of the produced raw image, e.g. xyztc.")
    parser.add_argument('--transition-classifier-file', dest='transition_classifier_filename', type=str,
                        default=None, help="Transition classifier filename, or None if distance-based energies should be used.")
    parser.add_argument('--transition-classifier-path', dest='transition_classifier_path', type=str, default='/')
    parser.add_argument('--out-ctc-file', type=str, dest='output_ctc', required=True, 
                        help='Filename of the new ctc res_track.txt')
    parser.add_argument('--trans-par', dest='trans_par', type=float, default=5.0,
                        help='alpha for the transition prior')
    parser.add_argument('--gap-treshold', dest='gap_treshold', type=float, default=0.05,
                        help='Threshold for the gap closing probability.')
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help='Turn on verbose logging', default=False)
    parser.add_argument('--plugin-paths', dest='pluginPaths', type=str, nargs='+',
                        default=[os.path.abspath('../hytra/plugins')],
                        help='A list of paths to search for plugins for the tracking pipeline.')
    args, _ = parser.parse_known_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    trackingGraph = JsonTrackingGraph(model_filename=args.model_filename, result_filename=args.result_filename)

    gap_closer = JsonGapCloser(trackingGraph,
        args.label_image_filename,
        args.label_image_path,
        args.raw_filename,
        args.raw_path,
        args.raw_axes,
        args.pluginPaths,
        args.verbose)
    gap_closer.run(
        args.input_ctc,
        args.output_ctc,
        args.transition_classifier_filename,
        args.transition_classifier_path
        args.gap_treshold)

