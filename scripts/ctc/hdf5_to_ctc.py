# pythonpath modification to make hytra and empryonic available 
# for import without requiring it to be installed
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('..'))
# standard imports
import configargparse as argparse
import numpy as np
import h5py
import vigra
import time
import glob
import logging
from skimage.external import tifffile

def get_num_frames(options):
    if len(options.input_files) == 1:
        with h5py.File(options.input_files[0], 'r') as in_h5:
            return in_h5[options.label_image_path].shape[0]
    else:
        return len(options.input_files)


def get_frame_label_image(timestep, options):
    if len(options.input_files) == 1:
        with h5py.File(options.input_files[0], 'r') as in_h5:
            return np.array(in_h5[options.label_image_path][timestep, ..., 0]).squeeze()
    else:
        with h5py.File(options.input_files[timestep], 'r') as in_h5:
            return np.array(in_h5[options.label_image_path]).squeeze()


def get_frame_dataset(timestep, dataset, options):
    if len(options.input_files) == 1:
        with h5py.File(options.input_files[0], 'r') as in_h5:
            ds_name = 'tracking/' + format(timestep, "0{}".format(options.h5group_zero_padding)) + '/' + dataset
            if ds_name in in_h5:
                return np.array(in_h5[ds_name])
    else:
        with h5py.File(options.input_files[timestep], 'r') as in_h5:
            ds_name = 'tracking/' + dataset
            if ds_name in in_h5:
                return np.array(in_h5[ds_name])

    return np.zeros(0)


def save_frame_to_tif(timestep, label_image, options):
    if len(options.input_files) == 1:
        filename = options.output_dir + '/man_track' + format(timestep, "0{}".format(options.filename_zero_padding)) + '.tif'
    else:
        filename = options.output_dir + '/mask' + format(timestep, "0{}".format(options.filename_zero_padding)) + '.tif'
    label_image = np.swapaxes(label_image, 0, 1)
    if len(label_image.shape) == 2: # 2d
        vigra.impex.writeImage(label_image.astype('uint16'), filename)
    else: # 3D
        label_image = np.transpose(label_image, axes=[2, 0, 1])
        tifffile.imsave(filename, label_image.astype('uint16'))



def save_tracks(tracks, num_frames, options):
    if len(options.input_files) == 1:
        filename = options.output_dir + '/man_track.txt'
    else:
        filename = options.output_dir + '/res_track.txt'
    with open(filename, 'wt') as f:
        for key, value in tracks.items():
            if len(value) == 2:
                value.append(num_frames - 1)
            # our track value contains parent, begin, end
            # but here we need begin, end, parent. so swap
            f.write("{} {} {} {}\n".format(key, value[1], value[2], value[0]))


def remap_label_image(label_image, mapping):
    """ 
    given a label image and a mapping, creates and 
    returns a new label image with remapped object pixel values 
    """
    remapped_label_image = np.zeros(label_image.shape, dtype=label_image.dtype)
    for dest, src in mapping.items():
        remapped_label_image[label_image == dest] = src

    return remapped_label_image


def convert_label_volume(options):
    num_frames = get_num_frames(options)
    if num_frames == 0:
        logging.getLogger('hdf5_to_ctc.py').error("Cannot work on empty set")
        return

    # for each track, indexed by first label, store [parent, begin, end]
    tracks = {}
    old_mapping = {} # mapping from label_id to track_id
    new_track_id = 1

    # handle frame 0 -> only add those nodes that are referenced from frame 1 events
    label_image = get_frame_label_image(0, options)
    label_image_indices = np.unique(label_image)
    logging.getLogger('hdf5_to_ctc.py').debug("Processing frame 0 of shape {}".format(label_image.shape))

    moves = get_frame_dataset(1, "Moves", options)
    splits = get_frame_dataset(1, "Splits", options)
    # splits could be empty
    if len(splits) == 0:
        if len(moves) == 0:
            referenced_labels = set([])
        else:
            referenced_labels = set(moves[:, 0])
    elif len(moves) == 0:
        referenced_labels = set(splits[:, 0])
    else:
        referenced_labels = set(moves[:, 0]) | set(splits[:, 0]) # set union

    for l in referenced_labels:
        if l == 0 or not l in label_image_indices:
            continue
        old_mapping[l] = new_track_id
        tracks[new_track_id] = [0, 0]
        new_track_id += 1
    remapped_label_image = remap_label_image(label_image, old_mapping)
    save_frame_to_tif(0, remapped_label_image, options)
    logging.getLogger('hdf5_to_ctc.py').debug("Tracks in first frame: {}".format(new_track_id))

    # handle all further frames by remapping their indices
    for frame in range(1, num_frames):
        old_label_image = label_image
        old_label_image_indices = np.unique(old_label_image)
        start_time = time.time()
        label_image = get_frame_label_image(frame, options)
        label_image_indices = np.unique(label_image)
        logging.getLogger('hdf5_to_ctc.py').debug("Processing frame {} of shape {}".format(frame, label_image.shape))
        mapping = {}

        moves = get_frame_dataset(frame, "Moves", options)
        splits = get_frame_dataset(frame, "Splits", options)
        
        # find the continued tracks
        for src, dest in moves:
            if src == 0 or dest == 0 or not src in old_label_image_indices or not dest in label_image_indices:
                continue
            # see whether this was a track continuation or the first leg of a new track
            if src in old_mapping.keys():
                mapping[dest] = old_mapping[src]
            elif len(splits)==0 or src not in list(splits[:,0]):
                mapping[dest] = new_track_id
                tracks[new_track_id] = [0, frame]
                new_track_id += 1

        # find all divisions
        for s in range(splits.shape[0]):
            # end parent track
            parent = splits[s, 0]

            if parent in old_mapping.keys():
                tracks[old_mapping[parent]].append(frame - 1)
            elif not parent in old_label_image_indices:
                logging.getLogger('hdf5_to_ctc.py').warning("Found division where parent id was not present in previous frame")
                parent = 0
                old_mapping[parent] = 0
            else:
                # insert a track of length 1 as parent of the new track
                old_mapping[parent] = new_track_id
                tracks[new_track_id] = [0, frame - 1, frame - 1]
                new_track_id += 1
                logging.getLogger('hdf5_to_ctc.py').warning("Adding single-node-track parent of division with id {}".format(new_track_id - 1))
                remapped_label_image = remap_label_image(old_label_image, old_mapping)
                save_frame_to_tif(frame-1, remapped_label_image, options)

            # create new tracks for all children
            for c in splits[s, 1:]:
                if c in label_image_indices:
                    tracks[new_track_id] = [old_mapping[parent], frame]
                    mapping[c] = new_track_id
                    new_track_id += 1
                else:
                    logging.getLogger('hdf5_to_ctc.py').warning("Discarding child {} of parent track {} because it is not present in image".format(c, parent))

        # find all tracks that ended (so not in a move or split (-> is parent))
        disappeared_indices = set(old_mapping.values()) - set(mapping.values())
        for idx in disappeared_indices:
            tracks[idx].append(frame - 1)

        # create a new label image with remapped indices (only those of tracks) and save it
        remapped_label_image = remap_label_image(label_image, mapping)
        save_frame_to_tif(frame, remapped_label_image, options)

        # save for next iteration
        old_mapping = mapping
        logging.getLogger('hdf5_to_ctc.py').debug("\tFrame done in {} secs".format(time.time() - start_time))
        logging.getLogger('hdf5_to_ctc.py').debug("Track count is now at {}".format(new_track_id))

    logging.getLogger('hdf5_to_ctc.py').info("Done processing frames, saving track info...")
    # done, save tracks
    save_tracks(tracks, num_frames, options)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert H5 event tracking solution to CTC format',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', is_config_file=True, help='config file path')

    parser.add_argument('--ctc-output-dir', type=str, dest='output_dir', required=True,
                        help='Folder where to save the label images starting with man_track00.tif, as well as a file man_track.txt')
    parser.add_argument('--h5-event-input-file-pattern', type=str, dest='input_file_pattern', required=True,
                        help='HDF5 file of ground truth, or file pattern matching individual frames')
    parser.add_argument('--h5-event-label-image-path', type=str, dest='label_image_path', default='label_image',
                        help='Path inside the HDF5 file(s) to the label image')
    parser.add_argument('--ctc-filename-zero-pad-length', type=int, dest='filename_zero_padding', default='3')
    parser.add_argument('--h5-group-zero-pad-length', type=int, dest='h5group_zero_padding', default='4')
    parser.add_argument("--verbose", dest='verbose', action='store_true', default=False)

    # parse command line
    args, unknown = parser.parse_known_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger('hdf5_to_ctc.py').debug("Ignoring unknown parameters: {}".format(unknown))

    # find all files matching the pattern
    args.input_files = glob.glob(args.input_file_pattern)
    args.input_files.sort()
    logging.info("Found {} files".format(len(args.input_files)))

    # make sure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    convert_label_volume(args)
