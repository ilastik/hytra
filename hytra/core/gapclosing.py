import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('..'))
# standard imports
import configargparse as argparse

import vigra
import time
import glob
from skimage.external import tifffile

import logging
import itertools
import h5py
import numpy as np
import commentjson as json
import networkx as nx
import copy
from hytra.pluginsystem.plugin_manager import TrackingPluginManager
import hytra.core.traxelstore as traxelstore
import hytra.core.jsongraph
from hytra.core.jsongraph import negLog, listify, JsonTrackingGraph
from  collections import OrderedDict

'''
This script is a edited compilation of `hdf5_to_ctc.py` and `mergerresolving.py`, edited and adapted for
gap closing (which has the purpose of linking tracks in order to overcome false positives or cells going
in and out of the field of view.)
'''

def getLogger():
    ''' logger to be used in this module '''
    return logging.getLogger(__name__)


def get_num_frames(input_files, label_image_path):
    if len(input_files) == 1:
        with h5py.File(input_files[0], 'r') as in_h5:
            return in_h5[label_image_path].shape[0]
    else:
        return len(input_files)


def get_frame_label_image(timestep, input_files, label_image_path):
    if len(input_files) == 1:
        with h5py.File(input_files[0], 'r') as in_h5:
            return np.array(in_h5[label_image_path][timestep, ..., 0]).squeeze()
    else:
        with h5py.File(input_files[timestep], 'r') as in_h5:
            return np.array(in_h5[label_image_path]).squeeze()


def get_frame_dataset(timestep, dataset, input_files):
    if len(input_files) == 1:
        with h5py.File(input_files[0], 'r') as in_h5:
            ds_name = 'tracking/' + format(timestep, "0{}".format(h5group_zero_padding)) + '/' + dataset
            if ds_name in in_h5:
                return np.array(in_h5[ds_name])
    else:
        with h5py.File(input_files[timestep], 'r') as in_h5:
            ds_name = 'tracking/' + dataset
            if ds_name in in_h5:
                return np.array(in_h5[ds_name])

    return np.zeros(0)


def save_tracks(tracks, num_frames, output_dir, input_files):
    if len(input_files) == 1:
        filename = output_dir + '/gap_closing.txt'
    else:
        filename = output_dir + '/gap_closing.txt'
    with open(filename, 'wt') as f:
        for key, value in tracks.iteritems():
            if len(value) == 2:
                value.append(num_frames - 1)
            # our track value contains parent, begin, end
            # but here we need begin, end, parent. so swap. Also not absolutely necessary.
            f.write("{} {} {} {}\n".format(key, value[1], value[2], value[0]))

def determine_tracks(output_dir, input_files, label_image_path, filename_zero_padding,
                        h5group_zero_padding):
    num_frames = get_num_frames(input_files, label_image_path)
    if num_frames == 0:
        logging.getLogger('hdf5_to_ctc.py').error("Cannot work on empty set")
        return

    # for each track, indexed by first label, store [parent, begin, end]
    tracks = {}
    old_mapping = {} # mapping from label_id to track_id
    new_track_id = 1

    # handle frame 0 -> only add those nodes that are referenced from frame 1 events
    label_image = get_frame_label_image(0, input_files, label_image_path)
    label_image_indices = np.unique(label_image)
    logging.getLogger('hdf5_to_ctc.py').debug("Processing frame 0 of shape {}".format(label_image.shape))

    moves = get_frame_dataset(1, "Moves", input_files)
    splits = get_frame_dataset(1, "Splits", input_files)
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
    logging.getLogger('hdf5_to_ctc.py').debug("Tracks in first frame: {}".format(new_track_id))

    # handle all further frames by remapping their indices
    for frame in range(1, num_frames):
        old_label_image = label_image
        old_label_image_indices = np.unique(old_label_image)
        start_time = time.time()
        label_image = get_frame_label_image(frame, input_files, label_image_path)
        label_image_indices = np.unique(label_image)
        logging.getLogger('hdf5_to_ctc.py').debug("Processing frame {} of shape {}".format(frame, label_image.shape))
        mapping = {}

        moves = get_frame_dataset(frame, "Moves", input_files)
        splits = get_frame_dataset(frame, "Splits", input_files)
        
        # find the continued tracks
        for src, dest in moves:
            if src == 0 or dest == 0 or not src in old_label_image_indices or not dest in label_image_indices:
                continue
            # see whether this was a track continuation or the first leg of a new track
            if src in old_mapping.keys():
                mapping[dest] = old_mapping[src]
            elif len(splits)==0 or src not in list(splits[:,0]):
                mapping[dest] = new_track_id
                tracks[new_track_id] = [0, (frame, new_track_id)]
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
                tracks[new_track_id] = [0, (frame - 1, new_track_id), frame - 1]
                new_track_id += 1
                logging.getLogger('hdf5_to_ctc.py').warning("Adding single-node-track parent of division with id {}".format(new_track_id - 1))

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
            label = list(old_mapping.keys())[list(old_mapping.values()).index(idx)]
            # tracks[idx].append(frame - 1)
            tracks[idx].append((frame - 1, label))

        # save for next iteration
        old_mapping = mapping
        logging.getLogger('hdf5_to_ctc.py').debug("\tFrame done in {} secs".format(time.time() - start_time))
        logging.getLogger('hdf5_to_ctc.py').debug("Track count is now at {}".format(new_track_id))

    logging.getLogger('hdf5_to_ctc.py').info("Done processing frames, saving track info...")
    # done, save tracks in `gap_closing.txt` for analysis. This is not imperative.
    save_tracks(tracks, num_frames, output_dir, input_files)
    return tracks



class GapCloser(object):
    def __init__(self, jsonTrackingGraph):
        # copy model and result because we will modify it here
        assert(isinstance(jsonTrackingGraph, JsonTrackingGraph))
        assert(jsonTrackingGraph.model is not None and len(jsonTrackingGraph.model) > 0)
        assert(jsonTrackingGraph.result is not None and len(jsonTrackingGraph.result) > 0)
        self.model = copy.copy(jsonTrackingGraph.model)
        self.result = copy.copy(jsonTrackingGraph.result)

        assert(self.result['detectionResults'] is not None)
        assert(self.result['linkingResults'] is not None)
        self.withDivisions = self.result['divisionResults'] is not None

        self.unresolvedGraph = None
        self.resolvedGraph = None
        self.resolvedJsonGraph = None

    def _determineTerminatingStartingTracks(self, tracks):
        """
        ** returns ** a list of detections (frame, id) sorted by frames to take into consideration for
        gap closing, which is tracks that end before time and ones that begin in the middle of the video.
        """

        # belle Ausgabe
        lookAt = []
        gapsExist = False
        gapsEnd = False
        # for gap closing we are not interested in the tracks ending in lastframe, that are no tuple
        for track in tracks:
            if type(tracks[track][-1]) != int:
                lookAt.append(tracks[track][-1]) # (frame,id)
                gapsEnd = True
            if type(tracks[track][1]) != int: # aici mai trebuie verificat cu bon datatset
                lookAt.append(tracks[track][1])
                gapsExist = True

        lookAt = sorted(lookAt, key=lambda x: x[0])

        # lookAt is not none only if tracks end before frames do and tracks begin in the middle of the video
        if gapsExist and gapsEnd:
            return lookAt

    def _createGraph(self, lookAt):
        """
        Set up a networkx graph consisting of
            - last detection in tracks that end before the video does
            - first detection of tracks beginning in the middle of the video

        ** returns ** the `unresolvedGraph`
        """
        self.unresolvedGraph = nx.DiGraph()
        def source(timestep, link):
            return int(timestep) - 1, link[0]

        def target(timestep, link):
            return int(timestep), link[1]

        def addNode(node):
            ''' add a node to the unresolved graph '''

            self.unresolvedGraph.add_node(node)

        # add nodes
        for node in lookAt:
            addNode(node)
            # find "frame-neighbours" to add connections to them
            for neighb in lookAt:
                if node != neighb and node[0] == neighb[0] + 1:
                    self.unresolvedGraph.add_edge(node, neighb)
                    print node, neighb, "edges"

        return self.unresolvedGraph

    def _getSegmentation(self,
                        resolvedGraph,
                        timesteps,
                        pluginManager,
                        label_image_filename,
                        label_image_path):
        '''
        Update segmentation of mergers (nodes in unresolvedGraph) from first timeframe to last
        and create new nodes in `resolvedGraph`. Links to merger nodes are duplicated to all new nodes.

        Uses the mergerResolver plugin to update the segmentations in the labelImages.
        '''

        imageProvider = pluginManager.getImageProvider()

        intTimesteps = [int(t) for t in timesteps]
        intTimesteps.sort()

        labelImages = {}

        for intT in intTimesteps:
            t = str(intT)
            # use image provider plugin to load labelimage
            labelImage = imageProvider.getLabelImageForFrame(
                label_image_filename, label_image_path, int(t))
            labelImages[t] = labelImage

        return labelImages

    def _computeObjectFeatures(self, labelImages,
                              pluginManager,
                              raw_filename,
                              raw_path,
                              raw_axes,
                              label_image_filename,
                              label_image_path,
                              resolvedGraph):
        """
        Computes object features for all nodes in the resolved graph because they
        are needed for the transition classifier or to compute new distances.

        **returns:** a dictionary of feature-dicts per node
        """
        rawImages = {}
        for t in labelImages.keys():
            rawImages[t] = pluginManager.getImageProvider().getImageDataAtTimeFrame(
                raw_filename, raw_path, raw_axes, int(t))

        getLogger().info("Computing object features")
        objectFeatures = {}
        imageShape = pluginManager.getImageProvider().getImageShape(label_image_filename, label_image_path)
        getLogger().info("Found image of shape {}".format(imageShape))
        # ndims = len(np.array(imageShape).squeeze()) - 1 # get rid of axes with length 1, and minus time axis
        # there is no time axis...
        ndims = len([i for i in imageShape if i != 1])
        getLogger().info("Data has dimensionality {}".format(ndims))
        for node in resolvedGraph.nodes_iter():
            intT, idx = node
            print intT, idx, "NODEE"
            # mask out this object only and compute features
            mask = labelImages[str(intT)].copy()
            mask[mask != idx] = 0
            mask[mask == idx] = 1

            # compute features, transform to one dict for frame
            frameFeatureDicts, ignoreNames = pluginManager.applyObjectFeatureComputationPlugins(
                ndims, rawImages[str(intT)], mask, intT, raw_filename)
            frameFeatureItems = []
            for f in frameFeatureDicts:
                frameFeatureItems = frameFeatureItems + f.items()
            frameFeatures = dict(frameFeatureItems)

            # extract all features for this one object
            objectFeatureDict = {}
            for k, v in frameFeatures.iteritems():
                if k in ignoreNames:
                    continue
                elif 'Polygon' in k:
                    objectFeatureDict[k] = v[1]
                else:
                    # print v
                    objectFeatureDict[k] = v[1, ...] # UNCOMMENT!
            objectFeatures[node] = objectFeatureDict

        return objectFeatures

    def minCostMaxFlowMergerResolving(self, resolvedGraph, objectFeatures, pluginManager, transitionClassifier=None, transitionParameter=5.0):
        """
        Find the optimal assignments within the `resolvedGraph` by running min-cost max-flow from the
        `dpct` module.

        Converts the `resolvedGraph` to our JSON model structure, predicts the transition probabilities 
        either using the given transitionClassifier, or using distance-based probabilities.

        **returns** a `nodeFlowMap` and `arcFlowMap` holding information on the usage of the respective nodes and links

        **Note:** cannot use `networkx` flow methods because they don't work with floating point weights.
        """

        trackingGraph = JsonTrackingGraph()
        for node in resolvedGraph.nodes_iter():
            additionalFeatures = {}
            if len(resolvedGraph.in_edges(node)) == 0:
                additionalFeatures['appearanceFeatures'] = [[0], [0]]
            if len(resolvedGraph.out_edges(node)) == 0:
                additionalFeatures['disappearanceFeatures'] = [[0], [0]]
            uuid = trackingGraph.addDetectionHypotheses([[0], [1]], **additionalFeatures)
            resolvedGraph.node[node]['id'] = uuid

        for edge in resolvedGraph.edges_iter():
            src = resolvedGraph.node[edge[0]]['id']
            dest = resolvedGraph.node[edge[1]]['id']

            featuresAtSrc = objectFeatures[edge[0]]
            featuresAtDest = objectFeatures[edge[1]]

            if transitionClassifier is not None:
                try:
                    featVec = pluginManager.applyTransitionFeatureVectorConstructionPlugins(
                        featuresAtSrc, featuresAtDest, transitionClassifier.selectedFeatures)
                except:
                    getLogger().error("Could not compute transition features of link {}->{}:".format(src, dest))
                    getLogger().error(featuresAtSrc)
                    getLogger().error(featuresAtDest)
                    raise
                featVec = np.expand_dims(np.array(featVec), axis=0)
                probs = transitionClassifier.predictProbabilities(featVec)[0]
            else:
                dist = np.linalg.norm(featuresAtDest['RegionCenter'] - featuresAtSrc['RegionCenter'])
                prob = np.exp(-dist / transitionParameter)
                probs = [1.0 - prob, prob]

            trackingGraph.addLinkingHypotheses(src, dest, listify(negLog(probs)))

        # track
        import dpct
        weights = {"weights": [1, 1, 1, 1]}
        mergerResult = dpct.trackMaxFlow(trackingGraph.model, weights)

        # transform results to dictionaries that can be indexed by id or (src,dest)
        nodeFlowMap = dict([(int(d['id']), int(d['value'])) for d in mergerResult['detectionResults']])
        arcFlowMap = dict([((int(l['src']), int(l['dest'])), int(l['value'])) for l in mergerResult['linkingResults']])

        return nodeFlowMap, arcFlowMap

    def _refineModel(self,
                     model, 
                     resolvedGraph,
                     uuidToTraxelMap, 
                     traxelIdPerTimestepToUniqueIdMap, 
                     mergerNodeFilter, 
                     mergerLinkFilter):
        """
        Take the `model` (JSON format) with mergers, remove the merger nodes, but add new
        de-merged nodes and links. Also updates `traxelIdPerTimestepToUniqueIdMap` locally and in the resulting file,
        such that the traxel IDs match the new connected component IDs in the refined images.

        `mergerNodeFilter` and `mergerLinkFilter` are methods that can filter merger detections
        and links from the respective lists in the `model` dict.
        
        **Returns** the updated `model` dictionary, which is the same as the input `model` (works in-place)
        """

        # remove merger detections
        model['segmentationHypotheses'] = filter(mergerNodeFilter, model['segmentationHypotheses'])

        # remove merger links
        model['linkingHypotheses'] = filter(mergerLinkFilter, model['linkingHypotheses'])

        # insert new nodes and update UUID to traxel map
        nextUuid = max(uuidToTraxelMap.keys()) + 1
        for node in self.unresolvedGraph.nodes_iter():
            if self.unresolvedGraph.node[node]['count'] > 1:
                newIds = self.unresolvedGraph.node[node]['newIds']
                del traxelIdPerTimestepToUniqueIdMap[str(node[0])][str(node[1])]
                for newId in newIds:
                    newDetection = {}
                    newDetection['id'] = nextUuid
                    newDetection['timestep'] = [node[0], node[0]]
                    model['segmentationHypotheses'].append(newDetection)
                    traxelIdPerTimestepToUniqueIdMap[str(node[0])][str(newId)] = nextUuid
                    nextUuid += 1

        # insert new links
        for edge in resolvedGraph.edges_iter():
            newLink = {}
            newLink['src'] = traxelIdPerTimestepToUniqueIdMap[str(edge[0][0])][str(edge[0][1])]
            newLink['dest'] = traxelIdPerTimestepToUniqueIdMap[str(edge[1][0])][str(edge[1][1])]
            model['linkingHypotheses'].append(newLink)

        # save
        return model

    def _refineResult(self,
                      result,
                      nodeFlowMap,
                      arcFlowMap,
                      resolvedGraph,
                      traxelIdPerTimestepToUniqueIdMap,
                      mergerNodeFilter,
                      mergerLinkFilter):
        """
        Update a `result` dict by removing the mergers and adding the refined nodes and links.

        Operates on a `result` dictionary in our JSON result style with mergers,
        the resolved and unresolved graph as well as
        the `nodeFlowMap` and `arcFlowMap` obtained by running tracking on the `resolvedGraph`.

        Updates the `result` dictionary so that all merger nodes are removed but the new nodes
        are contained with the appropriate links and values.

        `mergerNodeFilter` and `mergerLinkFilter` are methods that can filter merger detections
        and links from the respective lists in the `result` dict.

        **Returns** the updated `result` dict, which is the same as the input `result` (works in-place)
        """

        # filter merger edges
        result['detectionResults'] = [r for r in result['detectionResults'] if mergerNodeFilter(r)]
        result['linkingResults'] = [r for r in result['linkingResults'] if mergerLinkFilter(r)]

        # add new nodes
        for node in self.unresolvedGraph.nodes_iter():
            if self.unresolvedGraph.node[node]['count'] > 1:
                newIds = self.unresolvedGraph.node[node]['newIds']
                for newId in newIds:
                    uuid = traxelIdPerTimestepToUniqueIdMap[str(node[0])][str(newId)]
                    resolvedNode = (node[0], newId)
                    resolvedResultId = resolvedGraph.node[resolvedNode]['id']
                    newDetection = {'id': uuid, 'value': nodeFlowMap[resolvedResultId]}
                    result['detectionResults'].append(newDetection)

        # add new links
        for edge in resolvedGraph.edges_iter():
            newLink = {}
            newLink['src'] = traxelIdPerTimestepToUniqueIdMap[str(edge[0][0])][str(edge[0][1])]
            newLink['dest'] = traxelIdPerTimestepToUniqueIdMap[str(edge[1][0])][str(edge[1][1])]
            srcId = resolvedGraph.node[edge[0]]['id']
            destId = resolvedGraph.node[edge[1]]['id']
            newLink['value'] = arcFlowMap[(srcId, destId)]
            result['linkingResults'].append(newLink)

        return result


    # ------------------------------------------------------------
    def run(
            self,
            output_dir,
            input_files,
            filename_zero_padding,
            h5_event_label_image_path,
            h5group_zero_padding,
            label_image_filename,
            label_image_path,
            raw_filename,
            raw_path,
            raw_axes,
            out_label_image,
            pluginPaths,
            transition_classifier_filename,
            transition_classifier_path,
            verbose
        ):
        """
        Run merger resolving with the given configuration in `options`:

        1. find mergers in the given model and result
        2. build graph of the unresolved (merger) nodes and their direct neighbors
        3. use a mergerResolving plugin to refine the merger nodes and their segmentation
        4. run min-cost max-flow tracking to find the fate of all the de-merged objects
        5. export refined model, result, and segmentation

        """

        traxelIdPerTimestepToUniqueIdMap, uuidToTraxelMap = hytra.core.jsongraph.getMappingsBetweenUUIDsAndTraxels(self.model)
        timesteps = [t for t in traxelIdPerTimestepToUniqueIdMap.keys()]

        mergers, detections, links, divisions = hytra.core.jsongraph.getMergersDetectionsLinksDivisions(self.result, uuidToTraxelMap, self.withDivisions)

        # ------------------------------------------------------------

        pluginManager = TrackingPluginManager(verbose=verbose, pluginPaths=pluginPaths)
        pluginManager.setImageProvider('LocalImageLoader')

        # ------------------------------------------------------------

        mergersPerTimestep = hytra.core.jsongraph.getMergersPerTimestep(mergers, timesteps)
        linksPerTimestep = hytra.core.jsongraph.getLinksPerTimestep(links, timesteps)
        detectionsPerTimestep = hytra.core.jsongraph.getDetectionsPerTimestep(detections, timesteps)
        divisionsPerTimestep = hytra.core.jsongraph.getDivisionsPerTimestep(divisions, linksPerTimestep, timesteps, self.withDivisions)
        mergerLinks = hytra.core.jsongraph.getMergerLinks(linksPerTimestep, mergersPerTimestep, timesteps)
        
        # determine the early track terminations and the late track starting points
        tracks = determine_tracks(output_dir, input_files, h5_event_label_image_path, filename_zero_padding, h5group_zero_padding)
        lookAt = self._determineTerminatingStartingTracks(tracks)
        print lookAt, " if lookAt = None, nothing to be done, existing"

        if lookAt == None:
            getLogger().info("There are no gaps to close, nothing to be done. Writing the output...")
            # segmentation
            intTimesteps = [int(t) for t in timesteps]
            intTimesteps.sort()
            imageProvider = pluginManager.getImageProvider()
            h5py.File(out_label_image, 'w').close()
            for t in intTimesteps:
                labelImage = imageProvider.getLabelImageForFrame(label_image_filename, label_image_path, int(t))
                pluginManager.getImageProvider().exportLabelImage(labelImage, int(t), out_label_image, label_image_path)
        else:
            # set up unresolved graph and then refine the nodes to get the resolved graph
            resolvedGraph = self._createGraph(lookAt)
            labelImages = self._getSegmentation(resolvedGraph,
                                               timesteps,
                                               pluginManager,
                                               label_image_filename,
                                               label_image_path)

            # ------------------------------------------------------------
            # compute new object features
            objectFeatures = self._computeObjectFeatures(labelImages,
                                                         pluginManager,
                                                         raw_filename,
                                                         raw_path,
                                                         raw_axes,
                                                         label_image_filename,
                                                         label_image_path,
                                                         resolvedGraph)

            # ------------------------------------------------------------
            # load transition classifier if any
            if transition_classifier_filename is not None:
                getLogger().info("\tLoading transition classifier")
                transitionClassifier = traxelstore.RandomForestClassifier(
                    transition_classifier_path, transition_classifier_filename)
            else:
                getLogger().info("\tUsing distance based transition energies")
                transitionClassifier = None

            # ------------------------------------------------------------
            # run min-cost max-flow to find merger assignments
            getLogger().info("Running min-cost max-flow to find resolved merger assignments")

            nodeFlowMap, arcFlowMap = self.minCostMaxFlowMergerResolving(resolvedGraph, objectFeatures, pluginManager, transitionClassifier)

            # ------------------------------------------------------------
            # fuse results into a new solution

            # 1.) replace merger nodes in JSON graph by their replacements -> new JSON graph
            #     update UUID to traxel map.
            #     a) how do we deal with the smaller number of states? 
            #        Does it matter as we're done with tracking anyway..?

            def mergerNodeFilter(jsonNode):
                uuid = int(jsonNode['id'])
                traxels = uuidToTraxelMap[uuid]
                return not any(t[1] in mergersPerTimestep[str(t[0])] for t in traxels)

            def mergerLinkFilter(jsonLink):
                srcUuid = int(jsonLink['src'])
                destUuid = int(jsonLink['dest'])
                srcTraxels = uuidToTraxelMap[srcUuid]
                destTraxels = uuidToTraxelMap[destUuid]
                return not any((str(destT[0]), (srcT[1], destT[1])) in mergerLinks for srcT, destT in itertools.product(srcTraxels, destTraxels))

            self.model = self._refineModel(self.model,
                                           resolvedGraph,
                                           uuidToTraxelMap,
                                           traxelIdPerTimestepToUniqueIdMap,
                                           mergerNodeFilter,
                                           mergerLinkFilter)

            # 2.) new result = union(old result, resolved mergers) - old mergers

            self.result = self._refineResult(self.result,
                                             nodeFlowMap,
                                             arcFlowMap,
                                             resolvedGraph,
                                             traxelIdPerTimestepToUniqueIdMap,
                                             mergerNodeFilter,
                                             mergerLinkFilter)

            # 3.) export refined segmentation
            h5py.File(out_label_image, 'w').close()
            for t in timesteps:
                pluginManager.getImageProvider().exportLabelImage(labelImages[t], int(t), out_label_image, label_image_path)
