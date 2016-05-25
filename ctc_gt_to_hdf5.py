#!/usr/bin/env python
import configargparse as argparse
import os
import logging
import numpy as np
import h5py
from skimage.external import tifffile
import glob
import toolbox.util.axesconversion

def find_splits(filename, start_frame):
    # store split events indexed by timestep, then parent
    split_events = {}
    num_splits = 0
    # reads the man_track.txt file
    with open(filename, 'rt') as tracks:
        for line in tracks:
            track_start = map(int, line.split())
            if track_start[3] > 0:
                num_splits += 1
                # parent = 0 is appearance, otherwise a split
                parent = track_start[3]
                timestep = track_start[1] - start_frame
                idx = track_start[0]

                if not timestep in split_events:
                    split_events[timestep] = {}
                if not parent in split_events[timestep]:
                    split_events[timestep][parent] = []

                split_events[timestep][parent].append(idx)

    logging.info("Found number of splits: {}".format(num_splits))
    return split_events

def remap_label_image(label_image, mapping):
    """ 
    given a label image and a mapping, creates and 
    returns a new label image with remapped object pixel values 
    """
    remapped_label_image = np.zeros_like(label_image)
    for src, dest in mapping.iteritems():
        remapped_label_image[label_image == src] = dest

    return remapped_label_image

def remap_events(events, mappingA, mappingB=None):
    """
    Takes a numpy.ndarray of `events` and applies the specified mappings (dictionaries).
    Each row is mapped as follows: the first element is mapped according to `mappingA`,
    where all following elements are mapped with `mappingB`.

    Thus for divisions or moves, `mappingA` would refer to the previous frame, and `mappingB`
    to the current one.

    **Returns** the remapped events as numpy.ndarray
    """
    if len(events.shape) == 1:
        events = np.expand_dims(events, axis=1)

    new_events = np.zeros_like(events)
    # set up a lambda function to manage the mapping, then use numpy's vectorize to allow it to
    # be applied to every individual element in the vector.
    new_events[:, 0] = np.apply_along_axis(np.vectorize(lambda x: mappingA[x]), 0, events[:, 0])

    if events.shape[1] > 1:
        if mappingB is None:
            raise AssertionError("Need two sets of mappings for division and move events!")
        new_events[:, 1:] = np.apply_along_axis(np.vectorize(lambda x: mappingB[x]), 0, events[:, 1:])
    
    return new_events

def find_label_image_remapping(label_image):
    """
    Given any kind of label image, find a mapping that leaves the background at value
    zero, and all further object IDs such that they are consecutive and start at 1 (needed for VIGRA).

    **Returns** a dictionary that maps from label_image object ids to continuous labels, 
    or None if the labels are already good
    """
    # check whether we already have continuous labels
    labels = list(np.unique(label_image))
    labels.sort()
    if labels == list(range(max(labels))):
        return None

    # otherwise, create the appropriate mapping
    continuous_labels = list(range(len(labels)))
    return dict(zip(labels, continuous_labels))

def save_label_image_for_frame(options, label_volume, out_h5, frame, mapping_per_frame=None):
    """
    Takes a specific frame or all frames from the `label_volume` and stores them in `out_h5`.

    **If** `options.single_frames == True` then the respective part is taken from the label_volume
    and stored in `/segmentation/labels` of `out_h5`, and a `mapping_per_frame` is applied if given.
    **Otherwise** the full volume is saved when `frame == 0`, and it does nothing for all further
    frames.

    `mapping_per_frame` must be a dictionary, with frames as keys, and the values are then again dictionaries
    from the indices of objects in a frame of `label_volume` to the output indices.
    """
    if options.single_frames:
        out_label_volume = label_volume[..., frame, :]
        if options.index_remapping and mapping_per_frame is not None:
            out_label_volume = remap_label_image(out_label_volume, mapping_per_frame[frame])

        out_h5.create_dataset("segmentation/labels", data=out_label_volume, dtype='u2', compression='gzip')
    else:
        raise NotImplementedError
        out_label_volume = np.transpose(label_volume, axes=[2, 1, 0, 3])
        
        if options.index_remapping and mapping_per_frame is not None:
            # remap every frame in the volume individually
            for frame in range(label_volume.shape[2]):
                remapped_frame = remap_label_image(out_label_volume[..., frame, 0], mapping_per_frame[frame])
                out_label_volume[..., frame, 0] = remapped_frame

        out_h5.create_dataset("label_image", data=out_label_volume, dtype='u2', compression='gzip')
        out_label_volume = (out_label_volume.swapaxes(1, 2))[..., np.newaxis]
        out_h5.create_dataset("label_image_T", data=out_label_volume, dtype='u2', compression='gzip')

def create_label_volume(options):
    # read image
    # label_volume = vigra.impex.readVolume('/export/home/lparcala/Fluo-N2DH-SIM/01_GT/TRA/man_track000.tif')
    label_volume = tifffile.imread(options.input_tif)
    logging.info("Found dataset of size {}".format(label_volume.shape))
    label_volume = toolbox.util.axesconversion.adjustOrder(label_volume, options.tif_input_axes, 'xyztc')
    timeaxis = label_volume.shape[3]

    split_events = find_splits(options.input_track, options.start_frame)

    if not os.path.exists(options.output_file):
        #create new folder for gt files
        os.mkdir(options.output_file)
    else:
        # as h5py somehow appends the old file instead of overwriting, do it manually
        if os.path.isfile(options.output_file):
            os.remove(options.output_file)
        elif os.path.isdir(options.output_file):
            import shutil
            shutil.rmtree(options.output_file)
            os.mkdir(options.output_file)

    # store object ids per frame and generate mappings
    objects_per_frame = []
    mapping_per_frame = {}

    for frame in range(timeaxis):
        label_image = label_volume[..., frame, 0]
        mapping_per_frame[frame] = find_label_image_remapping(label_image)

    # handle frame zero
    if not options.single_frames:
        # one holistic volume file
        out_h5 = h5py.File(options.output_file, 'w')
        ids = out_h5.create_group('ids')
        tracking = out_h5.create_group('tracking')
        # create empty tracking group for first frame
        tracking_frame = tracking.create_group(format(0, "04"))
    else:
        frame = 0
        out_h5 = h5py.File(options.output_file + format(frame, "04") + '.h5', 'w')
        tracking_frame = out_h5.create_group('tracking')
    save_label_image_for_frame(options, label_volume, out_h5, 0, mapping_per_frame)

    for frame in range(timeaxis):
        label_image = label_volume[..., frame, 0]
        objects = np.unique(label_image)
        objects_per_frame.append(set(objects))
        if not options.single_frames:
            ids.create_dataset(format(frame, "04"), data=objects, dtype='u2')

    # handle all further frames
    for frame in range(1, timeaxis):
        if options.single_frames:
            out_h5 = h5py.File(options.output_file + format(frame, "04") + '.h5', 'w')
            tracking_frame = out_h5.create_group('tracking')
        else:
            tracking_frame = tracking.create_group(format(frame, "04"))
        save_label_image_for_frame(options, label_volume, out_h5, frame, mapping_per_frame)

        # intersect track id sets of both frames, and place moves in HDF5 file
        tracks_in_both_frames = objects_per_frame[frame - 1] & objects_per_frame[frame] - set([0])
        moves = zip(list(tracks_in_both_frames), list(tracks_in_both_frames))

        # add the found splits as both, mitosis and split events
        if frame in split_events.keys():
            splits_in_frame = split_events[frame]

            # make sure all splits have the same dimension
            splits = []
            for key, value in splits_in_frame.iteritems():
                value = [v for v in value if v in objects_per_frame[frame]]

                if key not in objects_per_frame[frame - 1]:
                    logging.warning("Parent {} of split is not in previous frame {}. Ignored".format(key, frame - 1))
                    continue

                if len(value) > 1:
                    if len(value) > 2:
                        logging.warning("Cutting off children of {} in timestep {}".format(key, frame))
                    # cut off divisions into more than 2
                    splits.append([key] + value[0:2])
                elif len(value) == 1:
                    # store as move
                    logging.warning("Store move ({},{}) instead of split into one in timestep {}".format(key, value[0], frame))
                    moves.append((key, value[0]))

            if len(splits) > 0:
                splits = np.array(splits)
                if options.index_remapping:
                    splits = remap_events(splits, mapping_per_frame[frame - 1], mapping_per_frame[frame])
                tracking_frame.create_dataset("Splits", data=splits, dtype='u2')
                mitosis = [splits[i][0] for i in range(splits.shape[0])]
                tracking_frame.create_dataset("Mitosis", data=np.array(mitosis), dtype='u2')

        if len(moves) > 0:
            if options.index_remapping:
                    moves = remap_events(np.array(moves), mapping_per_frame[frame - 1], mapping_per_frame[frame])
            tracking_frame.create_dataset("Moves", data=moves, dtype='u2')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert Cell Tracking Challenge Ground Truth to our HDF5 event format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-c', '--config', is_config_file=True, help='config file path')

    # file paths
    parser.add_argument('--ctc-track-input-tif', type=str, dest='tif_input_file_pattern', required=True,
                        help='File pattern of Cell Tracking Challenge data: man_track*.tif')
    parser.add_argument('--ctc-track-input-axes', type=str, dest='tif_input_axes', default='txy',
                        help='Axes string defining the input shape. Separate tiff files are stacked along the first axes.')
    parser.add_argument('--ctc-track-input-txt', type=str, dest='input_track', required=True,
                        help='Path to Cell Tracking Challenge manual tracking file: man_track.txt')
    parser.add_argument('--groundtruth', type=str, dest='output_file', required=True,
                        help='Filename for the resulting HDF5 file/folder.')
    parser.add_argument('--start-frame', type=int, dest='start_frame', default=0,
                        help='First frame number (usually 0, but e.g. their rapoport starts at 150')
    parser.add_argument('--ctc-to-gt-single-frames', action='store_true', dest='single_frames',
                        help='output single frame h5 files instead of one volume. Filename is appended with numbers.')
    parser.add_argument('--ctc-to-gt-index-remapping', action='store_true', dest='index_remapping',
                        help='Remap indices so that the objects in each frame have continuous ascending indices.')
    parser.add_argument("--verbose", dest='verbose', action='store_true', default=False)

    # parse command line
    options, unknown = parser.parse_known_args()
    options.input_tif = glob.glob(options.tif_input_file_pattern)
    options.input_tif.sort()

    if options.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    logging.debug("Ignoring unknown parameters: {}".format(unknown))

    create_label_volume(options)
