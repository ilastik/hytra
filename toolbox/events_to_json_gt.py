import sys

sys.path.append('../.')
sys.path.append('.')

from empryonic import io
import commentjson as json
import os
import argparse
import numpy as np
import h5py
from multiprocessing import Pool

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

def get_uuid_to_traxel_map(traxelIdPerTimestepToUniqueIdMap):
    timesteps = [t for t in traxelIdPerTimestepToUniqueIdMap.keys()]
    uuidToTraxelMap = {}
    for t in timesteps:
        for i in traxelIdPerTimestepToUniqueIdMap[t].keys():
            uuid = traxelIdPerTimestepToUniqueIdMap[t][i]
            if uuid not in uuidToTraxelMap:
                uuidToTraxelMap[uuid] = []
            uuidToTraxelMap[uuid].append((int(t), int(i)))

    # sort the list of traxels per UUID by their timesteps
    for v in uuidToTraxelMap.values():
        v.sort(key=lambda timestepIdTuple: timestepIdTuple[0])

    return uuidToTraxelMap

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Take a json file containing a result to a set of HDF5 events files',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', required=True, type=str, dest='model_filename',
                        help='Filename of the json model description')
    parser.add_argument('--input-files', type=str, nargs='+', dest='input_files', required=True,
                        help='HDF5 file of ground truth, or list of files for individual frames')
    parser.add_argument('--label-image-path', type=str, dest='label_image_path', default='label_image',
                        help='Path inside the HDF5 file(s) to the label image')
    parser.add_argument('--h5group-zero-pad-length', type=int, dest='h5group_zero_padding', default='4')
    parser.add_argument('--out', type=str, dest='out', required=True, help='Filename of the resulting json groundtruth')
    
    args = parser.parse_args()

    with open(args.model_filename, 'r') as f:
        model = json.load(f)

    # load forward mapping and create reverse mapping from json uuid to (timestep,ID)
    traxelIdPerTimestepToUniqueIdMap = model['traxelToUniqueId']
    uuidToTraxelMap = get_uuid_to_traxel_map(traxelIdPerTimestepToUniqueIdMap)

    # check that we have the proper number of incoming frames
    num_frames = get_num_frames(args)
    if num_frames == 0:
        print("Cannot work on empty set")
        sys.exit(0)

    # lists that store the events
    activeOutgoingLinks = {}
    activeIncomingLinks = {}
    objectCounts = {}
    activeDivisions = []

    # lists that store the JSON events
    detectionsJson = []
    divisionsJson = []
    linksJson = []

    # handle all frames by remapping their indices
    for frame in range(0, num_frames):
        # print("Processing frame {}".format(frame))

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

            # print("Found link {} -> {} ({}:{} -> {}:{})".format(s,t, frame-1, src, frame, dest))

            if s not in objectCounts:
                objectCounts[s] = 1
            if t not in objectCounts:
                objectCounts[t] = 1
            activeOutgoingLinks.setdefault(s, []).append(t)
            activeIncomingLinks.setdefault(t, []).append(s)

        # find all divisions and directly add them to json
        splits = get_frame_dataset(frame, "Splits", args)
        for parent, _, _ in splits:
            division = {}
            assert(str(parent) in traxelIdPerTimestepToUniqueIdMap[str(frame-1)])
            division['id'] = int(traxelIdPerTimestepToUniqueIdMap[str(frame-1)][str(parent)])
            activeDivisions.append(division['id'])
            division['value'] = True
            divisionsJson.append(division)

        # find all mergers (will store the same value in the same entry several times if this was a tracklet-merger)
        mergers = get_frame_dataset(frame, "Mergers", args)
        for obj, count in mergers:
            assert(str(obj) in traxelIdPerTimestepToUniqueIdMap[str(frame)])
            # print("Found merger {}: {} ({}:{})".format(traxelIdPerTimestepToUniqueIdMap[str(frame)][str(obj)], count, frame, obj))
            objectCounts[traxelIdPerTimestepToUniqueIdMap[str(frame)][str(obj)]] = count

        # add all object counts to JSON
        for obj, val in objectCounts.iteritems():
            detection = {}
            detection['id'] = int(obj)
            detection['value'] = int(val)
            detectionsJson.append(detection)

    def addLinkToJson(s,t,value):
        link = {}
        link['src'] = int(s)
        link['dest'] = int(t)
        link['value'] = int(value)
        linksJson.append(link)

    # figure out the values of links depending on the involved detections:
    # 1.) all active arcs with target detection count = 1 are easy, set and also decrease count values of targets
    for node, sources in activeIncomingLinks.iteritems():
        if objectCounts[node] == 1:
            assert(len(sources) <= 1)
            addLinkToJson(sources[0], node, 1)
            activeOutgoingLinks[sources[0]].remove(node)
            sources.remove(sources[0])
            # del objectCounts[node]
        elif len(sources) == 1 and objectCounts[sources[0]] == objectCounts[node]:
            # two mergers of same size have an active link in between
            addLinkToJson(sources[0], node, objectCounts[node])
            activeOutgoingLinks[sources[0]].remove(node)
            sources.remove(sources[0])
        elif len(sources) == objectCounts[node]:
            # merger forms from single nodes
            for s in sources:
                addLinkToJson(s, node, objectCounts[node])
                activeOutgoingLinks[s].remove(node)
            activeIncomingLinks[node] = []

    jsonRoot = {}
    jsonRoot['linkingResults'] = linksJson
    jsonRoot['detectionResults'] = detectionsJson
    jsonRoot['divisionResults'] = divisionsJson

    with open(args.out, 'w') as f:
        json.dump(jsonRoot, f, indent=4, separators=(',', ': '))



