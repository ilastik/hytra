import argparse
import os

import numpy as np
import h5py
import vigra


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

    print("Found splits: {}".format(num_splits))
    return split_events


def create_label_volume(options):
    # read image
    label_volume = vigra.impex.readVolume(options.input_tif)
    print("Found dataset of size {}".format(label_volume.shape))

    split_events = find_splits(options.input_track, options.start_frame)

    # as h5py somehow appends the old file instead of overwriting, do it manually
    if os.path.exists(options.output_file):
        os.remove(options.output_file)

    if not options.single_frames:
        # one holistic volume file
        out_h5 = h5py.File(options.output_file, 'w')
        out_label_volume = np.transpose(label_volume, axes=[2, 1, 0, 3])
        out_h5.create_dataset("label_image", data=out_label_volume, dtype='u2', compression='gzip')
        ids = out_h5.create_group('ids')
        tracking = out_h5.create_group('tracking')
        # create empty tracking group for first frame
        tracking_frame = tracking.create_group(format(0, "04"))
    else:
        frame = 0
        out_h5 = h5py.File(options.output_file + format(frame, "04") + '.h5', 'w')
        out_label_volume = label_volume[..., frame, :]
        out_label_volume = np.transpose(out_label_volume, axes=[1, 0, 2])
        out_h5.create_dataset("segmentation/labels", data=out_label_volume, dtype='u2', compression='gzip')
        tracking_frame = out_h5.create_group('tracking')


    # store object ids per frame
    objects_per_frame = []
    for frame in range(label_volume.shape[2]):
        objects = np.unique(label_volume[..., frame, 0])
        objects_per_frame.append(set(objects))
        if not options.single_frames:
            ids.create_dataset(format(frame, "04"), data=objects, dtype='u2')


    for frame in range(1, label_volume.shape[2]):
        if options.single_frames:
            out_h5 = h5py.File(options.output_file + format(frame, "04") + '.h5', 'w')
            out_label_volume = label_volume[..., frame, :]
            out_label_volume = np.transpose(out_label_volume, axes=[1, 0, 2])
            out_h5.create_dataset("segmentation/labels", data=out_label_volume, dtype='u2', compression='gzip')
            tracking_frame = out_h5.create_group('tracking')
        else:
            tracking_frame = tracking.create_group(format(frame, "04"))

        # intersect track id sets of both frames, and place moves in HDF5 file
        tracks_in_both_frames = objects_per_frame[frame - 1] & objects_per_frame[frame] - set([0])
        moves = zip(list(tracks_in_both_frames), list(tracks_in_both_frames))

        # add the found splits as both, mitosis and split events
        if frame in split_events.keys():
            splits_in_frame = split_events[frame]
            mitosis = splits_in_frame.keys()
            if len(mitosis) > 0:
                tracking_frame.create_dataset("Mitosis", data=np.array(mitosis), dtype='u2')

            # make sure all splits have the same dimension
            splits = []
            for key, value in splits_in_frame.iteritems():
                if len(value) > 1:
                    if len(value) > 2:
                        print("Cutting off children of {} in timestep {}".format(key, frame))
                    # cut off divisions into more than 2
                    splits.append([key] + value[0:2])
                elif len(value) == 1:
                    # store as move
                    print("Store move instead of split into one of {} in timestep {}".format(key, frame))
                    moves.append((key, value[0]))

            if len(splits) > 0:
                tracking_frame.create_dataset("Splits", data=np.array(splits), dtype='u2')

        if len(moves) > 0:
            tracking_frame.create_dataset("Moves", data=moves, dtype='u2')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert Cell Tracking Challenge Data to our HDF5')

    # file paths
    parser.add_argument('--input-tif', type=str, dest='input_tif', required=True,
                        help='First tif file of Cell Tracking Challenge data: man_track00.tif')
    parser.add_argument('--input-track', type=str, dest='input_track', required=True,
                        help='Path to Cell Tracking Challenge manual tracking file: man_track.txt')
    parser.add_argument('--output-file', type=str, dest='output_file', required=True,
                        help='Filename for the resulting HDF5 file.')
    parser.add_argument('--start-frame', type=int, dest='start_frame', default=0,
                        help='First frame number (usually 0, but e.g. their rapoport starts at 150')
    parser.add_argument('--single-frames', action='store_true', dest='single_frames',
                        help='output single frame h5 files instead of one volume. Filename is appended with numbers.')

    # parse command line
    args = parser.parse_args()

    create_label_volume(args)
