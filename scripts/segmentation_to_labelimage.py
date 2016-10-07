# pythonpath modification to make hytra and empryonic available 
# for import without requiring it to be installed
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
# standard imports
import h5py
import configargparse as argparse
import logging
import hytra.util.axesconversion
import vigra 
import numpy as np

def segmentation_to_hdf5(options):
    """
    The generated segmentation is one HDF5 dataset per timestep,
    and each of these datasets has shape 1(t),x,y,z,1(c).
    """
    in_h5 = h5py.File(options.segmentation_file, 'r')
    out_h5 = h5py.File(options.hdf5Path, 'w')
    
    data = in_h5[options.segmentation_path].value
    logging.getLogger('segmentation_to_labelimage.py').info("Found image of shape {}".format(data.shape))    
    data = hytra.util.axesconversion.adjustOrder(data, options.segmentation_axes, 'txyzc')
    logging.getLogger('segmentation_to_labelimage.py').info("Changed into shape {}".format(data.shape))

    for timeframe in range(data.shape[0]):
        logging.getLogger('segmentation_to_labelimage.py').info("Working on timestep {}".format(timeframe))
        internalPath = options.hdf5ImagePath % (timeframe, timeframe + 1, data.shape[1], data.shape[2], data.shape[3])
        frame = np.expand_dims(vigra.analysis.labelMultiArrayWithBackground(data[timeframe, ...].astype(np.uint32)), axis=0)
        out_h5.create_dataset(internalPath, data=frame, compression='gzip')

    logging.getLogger('segmentation_to_labelimage.py').info("Saved {} timeframes".format(data.shape[0]))

if __name__ == '__main__':
    """
    Convert an h5 segmentation volume to the HDF5 volume format as used in (old) ilastik versions 
    """

    parser = argparse.ArgumentParser(
        description='Convert segmentation tif files to hdf5',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', is_config_file=True, help='config file path')

    parser.add_argument('--in-segmentation-file', required=True, type=str, dest='segmentation_file',
                        help='Filename pattern of the all tif files containing the segmentation data')
    parser.add_argument('--in-segmentation-axes', type=str, dest='segmentation_axes', default='txyzc',
                        help='Axes string defining the input volume shape')
    parser.add_argument('--in-segmentation-path', type=str, dest='segmentation_path', default='exported_data',
                        help='Path inside the HDF5 file to the volume')
    parser.add_argument('--label-image-file', required=True, type=str, dest='hdf5Path',
                        help='filename of where the segmentation HDF5 file will be created')
    parser.add_argument('--label-image-path', type=str, dest='hdf5ImagePath',
                        help='Path inside ilastik project file to the label image',
                        default='/TrackingFeatureExtraction/LabelImage/0000/[[%d, 0, 0, 0, 0], [%d, %d, %d, %d, 1]]')
    parser.add_argument("--verbose", dest='verbose', action='store_true', default=False)

    # parse command line
    options, unknown = parser.parse_known_args()
    
    if options.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    logging.getLogger('segmentation_to_labelimage.py').debug("Ignoring unknown parameters: {}".format(unknown))

    segmentation_to_hdf5(options)
