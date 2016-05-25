import vigra
import configargparse as argparse
import logging
from skimage.external import tifffile
import toolbox.util.axesconversion
import glob
from toolbox.util.skimage_tifffile_hack import hack
import os

def convert_to_volume(options):
    # data = tifffile.imread(options.input_file)
    path, files = hack(options.input_file)
    os.chdir(path)
    data = tifffile.imread(files)
    reshapedData = toolbox.util.axesconversion.adjustOrder(data, options.tif_input_axes)

    print("Saving h5 volume of shape {}".format(data.shape))
    vigra.writeHDF5(reshapedData, options.output_file, options.output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert Cell Tracking Challenge raw data to a HDF5 stack',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', is_config_file=True, help='config file path')

    # file paths
    parser.add_argument('--ctc-raw-input-tif', type=str, dest='tif_input_file_pattern', required=True,
                        help='Filepattern matching all tif files of the Cell Tracking Challenge Dataset')
    parser.add_argument('--ctc-raw-input-axes', type=str, dest='tif_input_axes', default='txy',
                        help='Axes string defining the input shape. Separate tiff files are stacked along the first axes.')
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