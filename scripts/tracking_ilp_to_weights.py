# pythonpath modification to make hytra available 
# for import without requiring it to be installed
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
# standard imports
import h5py
import configargparse as argparse
import hytra.core.jsongraph

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract the weights from an ilastik tracking project.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', is_config_file=True, help='config file path', dest='config_file')
    parser.add_argument('--ilastik-tracking-project', required=True, type=str, dest='ilpFilename',
                        help='Filename of the ilastik tracking project')
    parser.add_argument('--weight-json-file', type=str, dest='out', required=True, help='Filename of the resulting weights JSON file')

    args, _ = parser.parse_known_args()

    # load settings
    with h5py.File(args.ilpFilename, 'r') as h5file:
        withDivisions = h5file['/ConservationTracking/Parameters/0000/withDivisions'].value
        transitionWeight = h5file['/ConservationTracking/Parameters/0000/transWeight'].value
        detectionWeight = 10.0
        divisionWeight = h5file['/ConservationTracking/Parameters/0000/divWeight'].value
        appearanceWeight = h5file['/ConservationTracking/Parameters/0000/appearanceCost'].value
        disappearanceWeight = h5file['/ConservationTracking/Parameters/0000/disappearanceCost'].value

    if withDivisions:
        weights = {'weights' : [transitionWeight, detectionWeight, divisionWeight, appearanceWeight, disappearanceWeight]}
    else:
        weights = {'weights' : [transitionWeight, detectionWeight, appearanceWeight, disappearanceWeight]}

    hytra.core.jsongraph.writeToFormattedJSON(args.out, weights)
