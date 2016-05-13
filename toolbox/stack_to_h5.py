import vigra
import configargparse as argparse
import logging
from vigra import numpy as np
from skimage.external import tifffile
import glob

def convert_to_volume(options):
    # data = vigra.impex.readVolume('/export/home/lparcala/Fluo-N2DH-SIM/01/t000.tif')
    data = tifffile.imread(options.input_file)
    if len(data.shape) == 3: # 2D
        data = np.expand_dims(data, axis=3)
    else:
        data = np.expand_dims(np.transpose(data, axes=[0, 3, 2, 1]), axis=4)
    print("Saving h5 volume of shape {}".format(data.shape))
    vigra.writeHDF5(data, options.output_file, options.output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert Cell Tracking Challenge raw data to a HDF5 stack',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', is_config_file=True, help='config file path')

    # file paths
    parser.add_argument('--ctc-raw-input-tif', type=str, dest='tif_input_file_pattern', required=True,
                        help='Filename of the first image of the tiff stack')
    parser.add_argument('--raw-data-file', type=str, dest='output_file', required=True,
                        help='Filename for the resulting HDF5 file.')
    parser.add_argument('--raw-data-path', type=str, dest='output_path', default='exported_data',
                        help='Path inside the HDF5 file to the data')
    parser.add_argument("--verbose", dest='verbose', action='store_true', default=False)

    # parse command line
    options, unknown = parser.parse_known_args()
    options.input_file = glob.glob(options.tif_input_file_pattern)
    options.input_file.sort()


    if options.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    logging.debug("Ignoring unknown parameters: {}".format(unknown))

    convert_to_volume(options)


# import vigra
# import numpy as np
# import argparse

# if __name__ == "__main__":
#         parser = argparse.ArgumentParser(description="Create a hdf5 stack from a series of tif images")
#         parser.add_argument('--prefix', dest='prefix', required=True, type=str, help='Prefix of filename before number component')
#         parser.add_argument('--extension', dest='extension', default='tif', type=str, help='Extension of filename (default = tif)')
#         parser.add_argument('--max-frame', dest='max_frame', type=int, required=True, help='Number of last frame to take into account (starts at 0, inclusive)')
#         parser.add_argument('--dims', dest='dims', type=int, default=2, help='Number of dimensions of the data, default=2')
#         parser.add_argument('--out_name', dest='out_name', default='stack.h5', type=str, help='Filename of output stack, default=stack.h5')
#         args = parser.parse_args()

#         name_prefix = args.prefix
#         extension = '.' + args.extension

#         max_frame_num = args.max_frame

#         if args.dims == 2:
#                 loadFunc = vigra.impex.readImage
#         else:
#                 loadFunc = vigra.impex.readVolume

#         d = loadFunc(name_prefix+'{:03}'.format(0)+extension)
#         d = np.expand_dims(d, axis=0)

#         for i in range(1,max_frame_num+1):
#             slice = loadFunc(name_prefix+'{:03}'.format(i)+extension)
#             slice = np.expand_dims(slice, axis=0)
#             d = np.append(d, slice, axis=0)
#             print("Added file: {}{:03}{}".format(name_prefix, i, extension))

#         vigra.impex.writeHDF5(d, args.out_name, 'data', compression='gzip')
#         print("Saved stack to {}".format(args.out_name))