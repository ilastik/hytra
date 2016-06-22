# pythonpath modification to make hytra available
# for import without requiring it to be installed
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
# standard importsfrom empryonic import io
import glob
import h5py
import numpy as np
import configargparse as argparse
import commentjson as json
import networkx as nx
import hytra.core.jsongraph

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
    parser = argparse.ArgumentParser(description='Project a set of HDF5 events files onto a json hypotheses graph and create a JSON groundtruth',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', is_config_file=True, help='config file path', dest='config_file')
    parser.add_argument('--graph-json-file', required=True, type=str, dest='model_filename',
                        help='Filename of the json model description')
    parser.add_argument('--h5-event-input-file-pattern', type=str, dest='input_file_pattern', required=True,
                        help='HDF5 file of ground truth, or pattern that matches GT files for individual frames')
    parser.add_argument('--label-image-path', type=str, dest='label_image_path', default='label_image',
                        help='Path inside the HDF5 file(s) to the label image (only needed if it is a single HDF5)')
    parser.add_argument('--h5group-zero-pad-length', type=int, dest='h5group_zero_padding', default='4')
    parser.add_argument('--out', type=str, dest='out', required=True, help='Filename of the resulting json groundtruth')

    args, _ = parser.parse_known_args()

    trackingGraph = hytra.core.jsongraph.JsonTrackingGraph(model_filename=args.model_filename)

    # load forward mapping and create reverse mapping from json uuid to (timestep,ID)
    traxelIdPerTimestepToUniqueIdMap = trackingGraph.traxelIdPerTimestepToUniqueIdMap
    uuidToTraxelMap = trackingGraph.uuidToTraxelMap

    args.input_files = glob.glob(args.input_file_pattern)
    args.input_files.sort()

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

            # ignore moves if they cannot be represented in the given model (due to missing edge in graph)
            found = False
            for link in trackingGraph.model['linkingHypotheses']:
                if link['src'] == s and link['dest'] == t:
                    found = True
            if not found:
                print("Did not find link {} to {}, ignoring it.".format(s, t))
                continue

            if s not in objectCounts:
                objectCounts[s] = 1
            if t not in objectCounts:
                objectCounts[t] = 1
            activeOutgoingLinks.setdefault(s, []).append(t)
            activeIncomingLinks.setdefault(t, []).append(s)

        # find all divisions and directly add them to json
        splits = get_frame_dataset(frame, "Splits", args)
        for parent, child1, child2 in splits:
            print("Found split of {} (t={}) into {} and {} ".format(parent, frame-1, child1, child2))
            assert(str(parent) in traxelIdPerTimestepToUniqueIdMap[str(frame-1)])
            parentUuid = traxelIdPerTimestepToUniqueIdMap[str(frame-1)][str(parent)]
            assert(objectCounts[parentUuid] > 0)

            for c in [child1, child2]:
                childUuid = traxelIdPerTimestepToUniqueIdMap[str(frame)][str(c)]
                if childUuid not in objectCounts:
                    objectCounts[childUuid] = 1
                activeOutgoingLinks.setdefault(parentUuid, []).append(childUuid)
                activeIncomingLinks.setdefault(childUuid, []).append(parentUuid)

            division = {}
            division['id'] = int(parentUuid)
            activeDivisions.append(division['id'])
            division['value'] = True
            divisionsJson.append(division)

        # find all mergers (will store the same value in the same entry several times if this was a tracklet-merger)
        mergers = get_frame_dataset(frame, "Mergers", args)
        for obj, count in mergers:
            assert(str(obj) in traxelIdPerTimestepToUniqueIdMap[str(frame)])
            # print("Found merger {}: {} ({}:{})".format(traxelIdPerTimestepToUniqueIdMap[str(frame)][str(obj)], count, frame, obj))
            objectCounts[traxelIdPerTimestepToUniqueIdMap[str(frame)][str(obj)]] = count

    maxCapacity = max(objectCounts.values())

    def addLinkToJson(s, t, value):
        link = {}
        link['src'] = int(s)
        link['dest'] = int(t)
        link['value'] = int(value)
        linksJson.append(link)

    # run min-cost max flow here!
    def resolveMergerArcFlows(maxCapacity):
        """
        The old ground truth is ambiguous when it comes to the number of targets going along each move arc.
        This is because arcs only get "active" labels, which is not enough when we have mergers.
        So we set up a graph where we first find the maximum flow along all the active arcs, and then
        look for the minimum cost configuration of doing so (including the original transition costs).
        This should correspond to the lowest energy configuration found by the ILP.
        """
        graph = nx.DiGraph()
        graph.add_node('sink')
        graph.add_node('source')

        def targetNodeName(n):
            return str(n) + '-V'
        def sourceNodeName(n):
            return str(n) + '-U'

        # add all nodes as two networkx nodes with an arc between them
        # that has the capacity of the detection's cell count
        for obj, val in objectCounts.iteritems():
            graph.add_node(sourceNodeName(obj))
            graph.add_node(targetNodeName(obj))
            graph.add_edge(sourceNodeName(obj), targetNodeName(obj), capacity=int(val), cost=0)

            # each node with no active incoming arcs gets connection to source
            if obj not in activeIncomingLinks or len(activeIncomingLinks[obj]) == 0:
                graph.add_edge('source', sourceNodeName(obj), capacity=int(val), cost=0)

            # each node without outgoing active arcs gets connection to sink
            if obj not in activeOutgoingLinks or len(activeOutgoingLinks[obj]) == 0:
                graph.add_edge(targetNodeName(obj), 'sink', capacity=int(val), cost=0)

            # each node with division gets an additional 1-capacity link from source to targetNodeName(obj)
            if obj in activeDivisions:
                assert(val == 1)
                graph.add_edge('source', targetNodeName(obj), capacity=1, cost=0)

        # add move edges with their transition cost (assume weight = 1 as we do not include any other costs)
        for target, sources in activeIncomingLinks.iteritems():
            for s in sources:
                cost = -1.0
                found = False
                for link in trackingGraph.model['linkingHypotheses']:
                    if link['src'] == s and link['dest'] == target:
                        # in the conservation tracking model, there is one feature per state (0,1,2,..) 
                        # and 1,2,... have the same value. So we take the delta between 0 and 1
                        # FIXME: min-cost flow below will use costs linearly (cost * flow along each arc), which is not
                        # what the Conservation Tracking Model does!
                        cost = link['features'][1][0] - link['features'][0][0]
                        found = True
                if not found:
                    print("Did not find link {} to {}, ignoring it.".format(s, target))
                    continue
                graph.add_edge(targetNodeName(s), sourceNodeName(target), capacity=maxCapacity, cost=cost)

        # find max flow that can fit through graph
        flowAmount,flowDict = nx.maximum_flow(graph, 'source', 'sink', capacity='capacity')
        print("Maximum Flow through GT graph is {}".format(flowAmount))

        # # run min cost max flow to disambiguate
        # graph['sink']['demand'] = flowAmount
        # graph['source']['demand'] = -flowAmount # negative means sending
        # # nodes without attribute demand are assumed to have a net flow of 0
        # flowCost, flowDict = nx.capacity_scaling(graph, demand='demand', capacity='capacity', weight='cost')

        # translate result back to our original edges
        for target, sources in activeIncomingLinks.iteritems():
            for s in sources:
                try:
                    addLinkToJson(s, target, flowDict[targetNodeName(s)][sourceNodeName(target)])
                except:
                    # this will fail if we encountered a link that is not present in the JSON graph, but we've already complained about that before
                    print("Did not add link {} -> {} to GT, ignoring it.".format(targetNodeName(s), sourceNodeName(target)))
                    pass

        # add all object counts to JSON, which have possibly been updated according to missing links in model
        for obj, val in objectCounts.iteritems():
            detection = {}
            detection['id'] = int(obj)
            fixedValue = flowDict[sourceNodeName(obj)][targetNodeName(obj)]
            detection['value'] = int(fixedValue)
            detectionsJson.append(detection)

    resolveMergerArcFlows(maxCapacity)

    jsonRoot = {}
    jsonRoot['linkingResults'] = linksJson
    jsonRoot['detectionResults'] = detectionsJson
    jsonRoot['divisionResults'] = divisionsJson

    with open(args.out, 'w') as f:
        json.dump(jsonRoot, f, indent=4, separators=(',', ': '))



