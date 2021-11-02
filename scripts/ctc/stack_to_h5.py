# pythonpath modification to make hytra and empryonic available 
# for import without requiring it to be installed
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('..'))
# standard imports
import vigra
import configargparse as argparse
import logging
import tifffile
import hytra.util.axesconversion
import glob
from hytra.util.skimage_tifffile_hack import hack

def convert_to_volume(options):
    # data = tifffile.imread(options.input_file)
    path, files = hack(options.input_file)
    logging.getLogger('stack_to_h5.py').debug("Found {} input tif files".format(len(files)))
    os.chdir(path)
    data = tifffile.imread(files)
    logging.getLogger('stack_to_h5.py').debug("Input shape is {}".format(data.shape))
    reshapedData = hytra.util.axesconversion.adjustOrder(data, options.tif_input_axes, options.output_axes)
    logging.getLogger('stack_to_h5.py').debug("Saving h5 volume of shape {} and dtype {}".format(reshapedData.shape, reshapedData.dtype))
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
    parser.add_argument("--raw-data-axes", dest='output_axes', type=str, default='txyzc',
                        help="axes ordering of the produced raw image, e.g. xyztc.")
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