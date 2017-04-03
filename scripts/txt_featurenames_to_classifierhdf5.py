from __future__ import print_function, absolute_import, nested_scopes, generators, division, with_statement, unicode_literals
import numpy as np
import h5py
import vigra
import configargparse as argparse
import logging
import glob
import os

def txt_to_hdf5(options):
    with open(options.txt_path + '/division_features.txt') as txt:
        content = txt.readlines()

    h5 = h5py.File(options.hdf5,'r+')
    division_path = h5['/DivisionDetection']
    # division_path.create_group['SelectedFeaturesCell Division features/']
    for i in range(len(content)):
        if i > len(content)-4: #Count, Mean and Variance
            grp = division_path.create_group('SelectedFeatures/Standard Object Features/' + content[i].strip())
        else:
            grp = division_path.create_group('SelectedFeatures/Cell Division Features/' + content[i].strip())


    with open(options.txt_path + '/object_count_features.txt') as txt:
        content = txt.readlines()

    countClassification_path = h5['/CountClassification']
    # division_path.create_group['SelectedFeaturesCell Division features/']
    for i in range(len(content)):
        countClassification_path.create_group('SelectedFeatures/Standard Object Features/' + content[i].strip())

    logging.info("Done writing features from txt into classifier")

if __name__ == '__main__':
    """
    Write the txt files that specify the features into the classifier.h5 file
    """

    parser = argparse.ArgumentParser(
        description='Convert txt feature files to hdf5',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', is_config_file=True, help='config file path')

    parser.add_argument('--txt-path', required=True, type=str, dest='txt_path',
                        help='directory of the .txt files')
    parser.add_argument('--classifier-file', required=True, type=str, dest='hdf5',
                        help='path/filename of the classifier')

    parser.add_argument("--verbose", dest='verbose', action='store_true', default=False)

    # parse command line
    options, unknown = parser.parse_known_args()
    # options.tif_input_files = glob.glob(options.tif_input_file_pattern)
    # options.tif_input_files.sort()

    if options.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    logging.debug("Ignoring unknown parameters: {}".format(unknown))

    try:
        txt_to_hdf5(options)
    except ValueError:
        logging.info("Selected Features already in the classifier file. Nothing to be done.")
