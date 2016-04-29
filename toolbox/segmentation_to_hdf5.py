import numpy as np
import h5py
import vigra
import argparse

def segmentation_to_hdf5(options):
    out_h5 = h5py.File(options.hdf5Path, 'w')
    for timeframe in range(len(options.tif_input_files)):
        data = vigra.impex.readImage(options.tif_input_files[timeframe], dtype='UINT16') # sure UINT32?
        internalPath = options.hdf5ImagePath % (timeframe, timeframe + 1, data.shape[0], data.shape[1], data.shape[2])
        out_h5.create_dataset(internalPath, data=data, dtype='u2', compression='gzip')


if __name__ == '__main__':
    """
    Convert the segmentation tif format to HDF5 volume
    """

    parser = argparse.ArgumentParser(description='Convert segmentation tif files to hdf5')
    parser.add_argument('--tif-input-files', required=True, type=str, nargs='+', dest='tif_input_files',
                        help='Filename of the tif file containing the segmentation data')
    parser.add_argument('--hdf5Path', required=True, type=str, dest='hdf5Path',
                        help='name and path to the hdf5 volume')
    parser.add_argument('--hdf5-image-path', type=str, dest='hdf5ImagePath',
                        help='Path inside ilastik project file to the label image',
                        default='/TrackingFeatureExtraction/LabelImage/0000/[[%d, 0, 0, 0, 0], [%d, %d, %d, %d, 1]]')

    # parse command line
    args = parser.parse_args()

    segmentation_to_hdf5(args)