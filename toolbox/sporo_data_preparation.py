import vigra
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Take two tiff files, one for the sporozite channel and one for the nucleus channel, \
            and create two files needed for further processing: a 3-channel hdf5 volume and a 1-channel nucleus HDF5.',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--sporo', required=True, type=str, dest='sporoFilename',
                        help='Filename of the sporozyte tiff')
    parser.add_argument('--nucleus', required=True, type=str, dest='nucleusFilename',
                        help='Filename of the nucleus tiff')
    parser.add_argument('--3channel-out', type=str, dest='threeChannelOut', required=True, help='Filename of the resulting 3 channel HDF5')
    parser.add_argument('--nucleus-channel-out', type=str, dest='nucleusChannelOut', required=True, help='Filename of the resulting nucleus channel HDF5')
    
    args = parser.parse_args()

    sporoChannel = vigra.impex.readVolume(args.sporoFilename)
    nucleusChannel = vigra.impex.readVolume(args.nucleusFilename)
    resultVolume = np.zeros((nucleusChannel.shape[0], nucleusChannel.shape[1], nucleusChannel.shape[2], 3), dtype='float32')
    resultVolume[...,1] = sporoChannel[...,0]
    resultVolume[...,2] = nucleusChannel[...,0]
    vigra.impex.writeHDF5(resultVolume, args.threeChannelOut, 'exported_data')
    vigra.impex.writeHDF5(nucleusChannel, args.nucleusChannelOut, 'exported_data')
