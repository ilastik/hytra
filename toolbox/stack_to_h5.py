import vigra
import argparse

def convert_to_volume(options):
    data = vigra.impex.readVolume(options.input_file)
    print("Saving h5 volume of shape {}".format(data.shape))
    vigra.writeHDF5(data, options.output_file, options.output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute TRA loss of a new labeling compared to ground truth')

    # file paths
    parser.add_argument('--input-file', type=str, dest='input_file', required=True,
                        help='Filename of the first image of the tiff stack')
    parser.add_argument('--output-file', type=str, dest='output_file', required=True,
                        help='Filename for the resulting HDF5 file.')
    parser.add_argument('--output-path', type=str, dest='output_path', default='exported_data',
                        help='Path inside the HDF5 file to the data')

    # parse command line
    args = parser.parse_args()

    convert_to_volume(args)