import argparse
import numpy as np
import h5py
import vigra
import os

def find_splits(filename):
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
                timestep = track_start[1]
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
    label_volume = vigra.impex.readVolume(options.input_dir + '/man_track000.tif')
    print("Found dataset of size {}".format(label_volume.shape))

    if not options.output_file:
        options.output_file = options.input_dir + '/ground_truth.h5'

    split_events = find_splits(options.input_dir + '/man_track.txt')

    # as h5py somehow appends the old file instead of overwriting, do it manually
    if os.path.exists(options.output_file):
        os.remove(options.output_file)

    with h5py.File(options.output_file, 'w') as out_h5:
        out_h5.create_dataset("label_image", data=label_volume, dtype='i', compression='gzip')
        ids = out_h5.create_group('ids')
        tracking = out_h5.create_group('tracking')

        # object ids per frame
        objects_per_frame = []
        for frame in range(label_volume.shape[2]):
            objects = np.unique(label_volume[..., frame, 0])
            ids.create_dataset(format(frame, "04"), data=objects, dtype='u2')
            objects_per_frame.append(set(objects))

        # move, mitosis and split events
        tracking_frame = tracking.create_group(format(0, "04"))
        for frame in range(1, label_volume.shape[2]):
            tracking_frame = tracking.create_group(format(frame, "04"))

            # intersect track id sets of both frames, and place moves in HDF5 file
            tracks_in_both_frames = objects_per_frame[frame - 1] & objects_per_frame[frame] - set([0])
            moves = np.array([list(tracks_in_both_frames), list(tracks_in_both_frames)]).transpose()
            tracking_frame.create_dataset("Moves", data=moves, dtype='u2')

            # add the found splits as both, mitosis and split events
            if frame in split_events.keys():
                splits_in_frame = split_events[frame]
                mitosis = splits_in_frame.keys()
                if len(mitosis) > 0:
                    tracking_frame.create_dataset("Mitosis", data=np.array(mitosis), dtype='u2')
                splits = [[key] + value for key, value in splits_in_frame.iteritems()]
		# make sure all splits have the same dimension
		max_split_length = max(map(len, splits))
		min_split_length = min(map(len, splits))
		if min_split_length != max_split_length:
			print("In timestep {}: Found splits longer than minimum {}, cutting off children to make number equal!".format(frame, min_split_length))
			for i, split in enumerate(splits):
				splits[i] = split[0:min_split_length]


                if len(splits) > 0:
                    tracking_frame.create_dataset("Splits", data=np.array(splits), dtype='u2')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute TRA loss of a new labeling compared to ground truth')

    # file paths
    parser.add_argument('--input-dir', type=str, dest='input_dir', required=True,
                        help='Folder which contains the label images starting with man_track00.tif, as well as a file man_track.txt')
    parser.add_argument('--output-file', type=str, dest='output_file',
                        help='Filename for the resulting HDF5 file. [default=$INPUT_DIR/ground_truth.h5]')

    # parse command line
    args = parser.parse_args()

    create_label_volume(args)
