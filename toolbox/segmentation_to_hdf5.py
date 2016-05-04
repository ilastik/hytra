import numpy as np
import h5py
import vigra
import configargparse as argparse
import logging
import glob
import os

def segmentation_to_hdf5(options):
    """
    The generated segmentation is one HDF5 dataset per timestep,
    and each of these datasets has shape 1(t),x,y,z,1(c).
    """
    out_h5 = h5py.File(options.hdf5Path, 'w')
    for timeframe in range(len(options.tif_input_files)):
        data = vigra.impex.readImage(options.tif_input_files[timeframe], dtype='UINT16') # sure UINT32?
        
        if timeframe == 0:
            logging.info("Found image of shape {}".format(data.shape))
        
        # put dimension in front for the time step
        data = np.expand_dims(data, axis=0)
        while len(data.shape) < 5:
            data = np.expand_dims(data, axis=-1)
        data = np.swapaxes(data, 1, 2)

        if timeframe == 0:
            logging.info("Changed into shape {}".format(data.shape))

        internalPath = options.hdf5ImagePath % (timeframe, timeframe + 1, data.shape[1], data.shape[2], data.shape[3])
        out_h5.create_dataset(internalPath, data=data, dtype='u2', compression='gzip')
        logging.info("Saved {} timeframes".format(timeframe))

if __name__ == '__main__':
    """
    Convert the segmentation tif format to HDF5 volume
    """

    parser = argparse.ArgumentParser(
        description='Convert segmentation tif files to hdf5',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', is_config_file=True, help='config file path')

    parser.add_argument('--ctc-segmentation-input-tif-pattern', required=True, type=str, dest='tif_input_file_pattern',
                        help='Filename pattern of the all tif files containing the segmentation data')
    parser.add_argument('--label-image-file', required=True, type=str, dest='hdf5Path',
                        help='filename of where the segmentation HDF5 file will be created')
    parser.add_argument('--label-image-path', type=str, dest='hdf5ImagePath',
                        help='Path inside ilastik project file to the label image',
                        default='/TrackingFeatureExtraction/LabelImage/0000/[[%d, 0, 0, 0, 0], [%d, %d, %d, %d, 1]]')
    parser.add_argument("--verbose", dest='verbose', action='store_true', default=False)

    # parse command line
    options, unknown = parser.parse_known_args()

    options.tif_input_files = glob.glob(options.tif_input_file_pattern)
    options.tif_input_files.sort()

    if options.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    logging.debug("Ignoring unknown parameters: {}".format(unknown))

    segmentation_to_hdf5(options)
