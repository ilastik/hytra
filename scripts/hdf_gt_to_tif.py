from __future__ import print_function
from __future__ import unicode_literals
############################### Convert hdf5 to file to tif #######################
# script can be used to e.g convert the gt of rapoport which is stored in hdf file
# into multiple tif files which then in turn can be evaluated using the evaluation
# script provided by the ctc.
#
# Example how to use:
# python hdf_gt_to_tif.py --output-dir ~/Documents/data/rapoport \ 
#  --input-file ~/Documents/data/rapoport/rapoport_ground_truth_sub_tolerant.h5 --label-image volume
###################################################################################

from builtins import range
import argparse
import numpy as np
import h5py
import vigra
import time

# major components where adapted form h5_to_ctc.py
def get_num_frames(options):
    if len(options.input_files) == 1:
        with h5py.File(options.input_files[0], 'r') as in_h5:
            return in_h5[options.label_image_path].shape[0]
    else:
        return len(options.input_files)


def save_frame_to_tif(timestep, label_image, options):
    if len(options.input_files) == 1:
        filename = options.output_dir + '/man_seg' + format(timestep, "0{}".format(options.filename_zero_padding)) + '.tif' # default, GIT
        # filename = options.output_dir + '/mask' + format(timestep, "0{}".format(options.filename_zero_padding)) + '.tif' # for letting SEGMEasure run before tracking
        # filename = options.output_dir + '/seg' + format(timestep, "0{}".format(options.filename_zero_padding)) + '.tif' # for converting Phillips prediction pams to segm
    else:
    	'This was not implemented here'
    vigra.impex.writeImage(label_image.astype('uint16'), filename)

def get_frame_label_image(timestep, options):
    if len(options.input_files) == 1:
        with h5py.File(options.input_files[0], 'r') as in_h5:
            return np.array(in_h5[options.label_image_path][timestep, ..., 0]).squeeze()
            # return np.array(in_h5[options.label_image_path][timestep, 0, ...]).squeeze()
    else:
        with h5py.File(options.input_files[timestep], 'r') as in_h5:
            return np.array(in_h5[options.label_image_path]).squeeze()

def convert_label_volume(options):
    num_frames = get_num_frames(options)
    if num_frames == 0:
        print("Cannot work on empty set")
        return
    print(num_frames)
    for frame in range(0, num_frames):
    	# print frame
    	label_image=get_frame_label_image(frame,options)
    	save_frame_to_tif(frame,label_image, options)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transform the input hdf5 file to single tifs. During this process ONLY the segmentation result (and not the tracking) is concidered.')

    # file paths
    parser.add_argument('--output-dir', type=str, dest='output_dir', required=True,
                        help='Folder where to save the label images starting with man_seg000.tif')
    parser.add_argument('--input-files', type=str, nargs='+', dest='input_files', required=True,
                        help='HDF5 file of ground truth, or list of files for individual frames')
    parser.add_argument('--label-image-path', type=str, dest='label_image_path', default='label_image',
                        help='Path inside the HDF5 file(s) to the label image')
    parser.add_argument('--filename-zero-pad-length', type=int, dest='filename_zero_padding', default='3')
    parser.add_argument('--h5group-zero-pad-length', type=int, dest='h5group_zero_padding', default='4')
    # parse command line
    args = parser.parse_args()

    convert_label_volume(args)

