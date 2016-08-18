# pythonpath modification to make hytra and empryonic available 
# for import without requiring it to be installed
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import skimage.external.tifffile as tiffile
import argparse
import numpy as np
import h5py
import hytra.util.axesconversion

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Take two tiff files, one for the sporozite channel and one for the nucleus channel, \
            and create two files needed for further processing: a 3-channel hdf5 volume and a 1-channel nucleus HDF5.',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--sporo', required=True, type=str, dest='sporoFilename',
                        help='Filename of the sporozyte tiff')
    parser.add_argument('--nucleus', required=True, type=str, dest='nucleusFilename',
                        help='Filename of the nucleus tiff')
    parser.add_argument('--input-axes', type=str, dest='inputAxes', default='txy',
                        help='Axes string defining the input volume shape')
    parser.add_argument('--output-axes', type=str, dest='outputAxes', default='txyc',
                        help='Axes string defining the output volume shape. The last one should be channels (c)')
    parser.add_argument('--3channel-out', type=str, dest='threeChannelOut', required=True, help='Filename of the resulting 3 channel HDF5')
    parser.add_argument('--nucleus-channel-out', type=str, dest='nucleusChannelOut', required=True, help='Filename of the resulting nucleus channel HDF5')
    parser.add_argument('--uint8', action='store_true', dest='use_uint8', help="Add this flag to force conversion to uint 8")
    parser.add_argument('--normalize', action='store_true', dest='normalize', help="Add this flag to force normalization between 0 and 1 if uint8 is not specified, 0 to 255 otherwise")

    args = parser.parse_args()

    # read data
    sporoChannel = tiffile.imread(args.sporoFilename)
    nucleusChannel = tiffile.imread(args.nucleusFilename)
    print("Found input images of dimensions {} and {}".format(sporoChannel.shape, nucleusChannel.shape))

    if args.normalize:
        # normalize to range 0-1
        sporoChannel = (sporoChannel-np.min(sporoChannel))/(np.max(sporoChannel)-np.min(sporoChannel))
        nucleusChannel = (nucleusChannel-np.min(sporoChannel))/(np.max(nucleusChannel)-np.min(sporoChannel))

    # adjust axes
    sporoChannel = hytra.util.axesconversion.adjustOrder(sporoChannel, args.inputAxes, args.outputAxes)
    nucleusChannel = hytra.util.axesconversion.adjustOrder(nucleusChannel, args.inputAxes, args.outputAxes)

    desiredType = 'float32'
    if args.use_uint8:
        desiredType = 'uint8'
        if args.normalize:
            sporoChannel *= 255
            nucleusChannel *= 255
    
    resultVolume = np.zeros((nucleusChannel.shape[0], nucleusChannel.shape[1], nucleusChannel.shape[2], 3), dtype=desiredType)
    resultVolume[...,1] = sporoChannel[...,0]
    resultVolume[...,2] = nucleusChannel[...,0]

    print("3channel out now has min {}, max {} and dtype {}".format(resultVolume.min(), resultVolume.max(), resultVolume.dtype))
    print("resulting shape: {}".format(resultVolume.shape))

    with h5py.File(args.threeChannelOut, 'w') as outH5:
        outH5.create_dataset(name='exported_data', data=resultVolume, dtype=desiredType)
    
    with h5py.File(args.nucleusChannelOut, 'w') as outH5:
        outH5.create_dataset(name='exported_data', data=nucleusChannel, dtype=desiredType)
