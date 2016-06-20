# pythonpath modification to make hytra available 
# for import without requiring it to be installed
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
# standard importsfrom empryonic import io
import commentjson as json
import argparse
import numpy as np
import h5py
from multiprocessing import Pool
from hytra.util.progressbar import ProgressBar

def get_num_frames(options):
    if len(options.input_files) == 1:
        with h5py.File(options.input_files[0], 'r') as in_h5:
            return in_h5[options.label_image_path].shape[0]
    else:
        return len(options.input_files)

def get_frame_dataset(timestep, dataset, options):
    if len(options.input_files) == 1:
        with h5py.File(options.input_files[0], 'r') as in_h5:
            ds_name = 'tracking/' + format(timestep, "0{}".format(options.h5group_zero_padding)) + '/' + dataset
            if ds_name in in_h5:
                return np.array(in_h5[ds_name])
    else:
        # print("Reading {}/{}".format(options.input_files[timestep], dataset))
        with h5py.File(options.input_files[timestep], 'r') as in_h5:
            ds_name = 'tracking/' + dataset
            if ds_name in in_h5:
                return np.array(in_h5[ds_name])

    return np.zeros(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Given a JSON graph and the HDF5 ground truth, check how often the ObjectCountClassifier is right',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', required=True, type=str, dest='model_filename',
                        help='Filename of the json model description')
    parser.add_argument('--input-files', type=str, nargs='+', dest='input_files', required=True,
                        help='HDF5 file of ground truth, or list of files for individual frames')
    parser.add_argument('--label-image-path', type=str, dest='label_image_path', default='label_image',
                        help='Path inside the HDF5 file(s) to the label image')
    parser.add_argument('--h5group-zero-pad-length', type=int, dest='h5group_zero_padding', default='4')
    
    args = parser.parse_args()

    print("Loading model...")
    with open(args.model_filename, 'r') as f:
        model = json.load(f)

    # load forward mapping and create reverse mapping from json uuid to (timestep,ID)
    traxelIdPerTimestepToUniqueIdMap = model['traxelToUniqueId']

    # check that we have the proper number of incoming frames
    num_frames = get_num_frames(args)
    if num_frames == 0:
        print("Cannot work on empty set")
        sys.exit(0)

    def getPredictedCount(uuid):
        segHyps = model['segmentationHypotheses']
        for i, s in enumerate(segHyps):
            if s['id'] == uuid:
                break
        if i == len(segHyps) and s['id'] != uuid:
            raise InvalidArgumentException

        feats = segHyps[i]['features']
        feats = np.array(feats)
        return np.argmin(feats)

    objectCounts = {}
    print("Extracting GT mergers...")
    progressBar = ProgressBar(stop=num_frames)
    progressBar.show(0)
    duplicates = 0
    # handle all frames by remapping their indices
    for frame in range(0, num_frames):
        # find moves, but only store in temporary list because 
        # we don't know the number of cells in that move yet
        moves = get_frame_dataset(frame, "Moves", args)
        for src, dest in moves:
            if src == 0 or dest == 0:
                continue
            
            assert(str(src) in traxelIdPerTimestepToUniqueIdMap[str(frame-1)])
            assert(str(dest) in traxelIdPerTimestepToUniqueIdMap[str(frame)])
            s = traxelIdPerTimestepToUniqueIdMap[str(frame-1)][str(src)]
            t = traxelIdPerTimestepToUniqueIdMap[str(frame)][str(dest)]

            # ignore moves within a tracklet, as it is contracted in JSON
            if s == t:
                continue

            if s not in objectCounts:
                objectCounts[s] = 1
            if t not in objectCounts:
                objectCounts[t] = 1

        # find all mergers
        mergers = get_frame_dataset(frame, "Mergers", args)
        for obj, gtCount in mergers:
            assert(str(obj) in traxelIdPerTimestepToUniqueIdMap[str(frame)])
            uuid = traxelIdPerTimestepToUniqueIdMap[str(frame)][str(obj)]
            if uuid in objectCounts and objectCounts[uuid] > 1:
                duplicates +=1
            objectCounts[uuid] = gtCount
        progressBar.show()

    # compute errors
    print("Evaluating predictions")
    progressBar = ProgressBar(stop=len(objectCounts))
    numMergers = 0
    truePositives = 0
    squaredError = 0.0
    for obj, gtCount in objectCounts.iteritems():
        predictedObjCount = getPredictedCount(obj)
        squaredError += (gtCount - predictedObjCount)**2

        if gtCount > 1:
            numMergers += 1
            if predictedObjCount == gtCount:
                truePositives += 1
        progressBar.show()

    mse = float(squaredError) / len(objectCounts)
    print("Found {} TP in {} mergers: {} % ({} duplicates)".format(truePositives, numMergers, float(truePositives)/numMergers, duplicates))
    print("Overall MSE: {} on {} objects".format(mse, len(objectCounts)))


