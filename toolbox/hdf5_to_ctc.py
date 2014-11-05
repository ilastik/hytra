import argparse
import numpy as np
import h5py
import vigra
import time


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
            ds_name = 'tracking/' + format(timestep, "04") + '/' + dataset
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
        filename = options.output_dir + '/man_track' + format(timestep, "03") + '.tif'
    else:
        filename = options.output_dir + '/mask' + format(timestep, "03") + '.tif'
    vigra.impex.writeVolume(label_image, filename, '', dtype='UINT16')


def save_tracks(tracks, num_frames, options):
    if len(options.input_files) == 1:
        filename = options.output_dir + '/man_track.txt'
    else:
        filename = options.output_dir + '/res_track.txt'
    with open(filename, 'wt') as f:
        for key, value in tracks.iteritems():
            if len(value) == 2:
                value.append(num_frames)
            # our track value contains parent, begin, end
            # but here we need begin, end, parent. so swap
            f.write("{} {} {} {}\n".format(key, value[1], value[2], value[0]))


def convert_label_volume(options):
    num_frames = get_num_frames(options)
    if num_frames == 0:
        print("Cannot work on empty set")
        return

    # for each track, indexed by first label, store [parent, begin, end]
    tracks = {}
    old_mapping = {}
    new_track_id = 1

    # handle frame 0 -> nothing to do yet
    label_image = get_frame_label_image(0, options)
    print("Processing frame 0 of shape {}".format(label_image.shape))
    save_frame_to_tif(0, label_image, options)

    # handle all further frames by remapping their indices
    for frame in range(1, num_frames):
        start_time = time.time()
        label_image = get_frame_label_image(frame, options)
        print("Processing frame {} of shape {}".format(frame, label_image.shape))
        mapping = {}

        # find the continued tracks
        moves = get_frame_dataset(frame, "Moves", options)
        for src, dest in moves:
            if src == 0 or dest == 0:
                continue
            # see whether this was a track continuation or the first leg of a new track
            if src in old_mapping.keys():
                mapping[dest] = old_mapping[src]
            else:
                mapping[dest] = new_track_id
                tracks[new_track_id] = [0, frame - 1]
                new_track_id += 1

        # find all divisions
        splits = get_frame_dataset(frame, "Splits", options)
        parents = []
        children = []

        for s in range(splits.shape[0]):
            # end parent track
            parent = splits[s, 0]
            parents.append(parent)

            if parent in old_mapping.keys():
                tracks[old_mapping[parent]].append(frame - 1)
            else:
                # insert a track of length 1 as parent of the new track
                mapping[parent] = new_track_id
                tracks[new_track_id] = [0, frame - 1, frame - 1]
                new_track_id += 1

            # create new tracks for all children
            for c in splits[s, 1:]:
                tracks[new_track_id] = [parent, frame]
                mapping[c] = new_track_id
                new_track_id += 1
                children.append(c)

        # find all tracks that ended (so not in a move or split (-> is parent))
        disappeared_indices = set(old_mapping.keys()) - set(mapping.keys())
        for idx in disappeared_indices:
            tracks[old_mapping[idx]].append(frame - 1)

        # create a new label image with remapped indices (only those of tracks) and save it
        remapped_label_image = np.zeros(label_image.shape, dtype=label_image.dtype)
        for dest, src in mapping.iteritems():
            remapped_label_image[label_image == dest] = src
        save_frame_to_tif(frame, remapped_label_image, options)

        # save for next iteration
        old_mapping = mapping
        print("\tFrame done in {} secs".format(time.time() - start_time))

    print("Done processing frames, saving track info...")
    # done, save tracks
    save_tracks(tracks, num_frames, options)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute TRA loss of a new labeling compared to ground truth')

    # file paths
    parser.add_argument('--output-dir', type=str, dest='output_dir', required=True,
                        help='Folder where to save the label images starting with man_track00.tif, as well as a file man_track.txt')
    parser.add_argument('--input-files', type=str, nargs='+', dest='input_files', required=True,
                        help='HDF5 file of ground truth, or list of files for individual frames')
    parser.add_argument('--label-image-path', type=str, dest='label_image_path', default='label_image',
                        help='Path inside the HDF5 file(s) to the label image')

    # parse command line
    args = parser.parse_args()

    convert_label_volume(args)