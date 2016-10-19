import logging
import itertools
import h5py
import os
import numpy as np
import networkx as nx
from hytra.pluginsystem.plugin_manager import TrackingPluginManager
import hytra.core.probabilitygenerator as probabilitygenerator
import hytra.core.jsongraph
from hytra.core.jsongraph import negLog, listify, JsonTrackingGraph


def getLogger():
    ''' logger to be used in this module '''
    return logging.getLogger(__name__)

class GapCloser(object):
    """
    Base class for all gap closing implementations. Use one of the derived classes
    that handle reading/writing data to the respective sources.
    """

    def __init__(self, pluginPaths=[os.path.abspath('../hytra/plugins')], verbose=False):
        self.Graph = None
        self.pluginManager = TrackingPluginManager(verbose=verbose, pluginPaths=pluginPaths)

        # should be filled by constructors of derived classes!
        self.model = None
        self.result = None


    def _determineTerminatingStartingTracks(self, input_ctc):
        """
        ** returns ** a list of detections (frame, id) sorted by frames to take into consideration for
        gap closing, which are tracks that end before time and ones that begin in the middle of the video.
        """
        # ctc format is: (id, start_frame, end_frame, parent_id)
        with open(input_ctc) as txt:
            content = txt.readlines()
        track_info = []
        for line in content:
            strings = line.strip('\n').split()
            track_info.append(tuple([int(t) for t in strings]))

        lookAt = []
        lastframe = max(track_info, key=lambda item:item[2])[2]
        parents = [t[-1] for t in track_info]

        for track in track_info:
            idx, start_frame, end_frame, parent_id = track
            if start_frame !=0 and parent_id == 0: # track started, but not at the beginning of the video, dah!
                lookAt.append((start_frame, idx, 'starting'))
            if end_frame != lastframe and idx not in parents:
                lookAt.append((end_frame, idx, 'ending')) # track ends AND has no continuation!

        return sorted(lookAt, key=lambda item:item[0])

    def _createGraph(self, lookAt):
        """
        Set up a networkx graph consisting of
            - last detection in tracks that end before the video does
            - first detection of tracks beginning in the middle of the video

        ** returns ** the `unGraph`
        """
        self.Graph = nx.DiGraph()
        def source(timestep, link):
            return int(timestep) - 1, link[0]

        def target(timestep, link):
            return int(timestep), link[1]

        def addNode(node):
            ''' add a node to the unresolved graph'''
            self.Graph.add_node((node[0], node[1]))

        # add nodes
        for node in lookAt:
            # find "frame-neighbours" to add connections to them
            for neighb in lookAt:
                if node != neighb and node[0] == neighb[0] - 2 and node[2] == 'ending' and neighb[2] == 'starting':
                    addNode(node)
                    addNode(neighb)
                    self.Graph.add_edge((node[0], node[1]), (neighb[0], neighb[1]))
                    getLogger().debug("Adding edges {}->{}".format(node, neighb))

        return self.Graph

    def _readLabelImage(self, timeframe):
        '''
        Should return the labelimage for the given timeframe
        '''
        raise NotImplementedError()

    def _thresholdGapClosing(self, objectFeatures, transitionClassifier=None, transitionParameter=5.0, threshold=0.05):
        """
        Find the optimal assignments within the `Graph` by looking at the transition porbability and
        deciding wheter it is above the `threshold` or not.

        Converts the `Graph` to our JSON model structure, predicts the transition probabilities 
        either using the given transitionClassifier, or using distance-based probabilities.

        **returns** a `ctc_arcFlowMap` holding information on the usage of the respective nodes and links
        """

        map_uuid_to_ctc = {}
        trackingGraph = JsonTrackingGraph()
        for node in self.Graph.nodes_iter():
            additionalFeatures = {}
            if len(self.Graph.in_edges(node)) == 0:
                additionalFeatures['appearanceFeatures'] = [[0], [0]]
            if len(self.Graph.out_edges(node)) == 0:
                additionalFeatures['disappearanceFeatures'] = [[0], [0]]
            uuid = trackingGraph.addDetectionHypotheses([[0], [1]], **additionalFeatures)
            self.Graph.node[node]['id'] = uuid
            map_uuid_to_ctc[uuid] = node

        ctc_arcFlowMap = {}
        for edge in self.Graph.edges_iter():
            src = self.Graph.node[edge[0]]['id']
            dest = self.Graph.node[edge[1]]['id']

            featuresAtSrc = objectFeatures[edge[0]]
            featuresAtDest = objectFeatures[edge[1]]

            if transitionClassifier is not None:
                try:
                    featVec = self.pluginManager.applyTransitionFeatureVectorConstructionPlugins(
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

            getLogger().debug("Looking at probabilities {} on edge {}:".format(probs, edge))

            if probs[1] > threshold:
                ctc_arcFlowMap[edge] = 1
            else:
                ctc_arcFlowMap[edge] = 0

        return ctc_arcFlowMap

    def _saveTracks(self, ctc_arcFlowMap, input_ctc, output_ctc):
        """
        Update the ctc result format (res_track.txt) by adding the right parents to the closed gaps.
        So read in the old sresult txtx file and write a new one.
        """

        if ctc_arcFlowMap is not None:
            with open(input_ctc) as input_txt:
                content = input_txt.readlines()

            with open(output_ctc, 'wt') as f:
                for line in content:
                    for edge in ctc_arcFlowMap:
                        strings = line.strip('\n').split()
                        track_info = tuple([int(t) for t in strings])
                        idx, start_frame, end_frame, parent_id = track_info
                        if edge[1][1] == idx and ctc_arcFlowMap[edge] == 1:
                            parent_id = edge[0][1]
                            line ="{} {} {} {}\n".format(idx, start_frame, end_frame, parent_id)
                    f.write(line)

    def _computeObjectFeatures(self, labelImages):
        '''
        Return the features per object as nested dictionaries:
        { (int(Timestep), int(Id)):{ "FeatureName" : np.array(value), "NextFeature": ...} }
        '''
        pass

    # ------------------------------------------------------------
    def run(self, input_ctc, output_ctc, transition_classifier_filename=None, transition_classifier_path=None, threshold=0.05):
        """
        Run merger resolving

        1. find mergers in the given model and result
        2. build graph of the unresolved (merger) nodes and their direct neighbors
        3. use a mergerResolving plugin to refine the merger nodes and their segmentation
        4. run min-cost max-flow tracking to find the fate of all the de-merged objects
        5. export refined segmentation, update member variables `model` and `result`

        """

        traxelIdPerTimestepToUniqueIdMap, uuidToTraxelMap = hytra.core.jsongraph.getMappingsBetweenUUIDsAndTraxels(self.model)
        timesteps = [t for t in traxelIdPerTimestepToUniqueIdMap.keys()]

        mergers, detections, links, divisions = hytra.core.jsongraph.getMergersDetectionsLinksDivisions(self.result, uuidToTraxelMap)
        lookAt = self._determineTerminatingStartingTracks(input_ctc)

        # ------------------------------------------------------------
        if lookAt == None:
            getLogger().info("There are no gaps to close, nothing to be done. Writing the output...")
        else:
            # set up unresolved graph and then refine the nodes to get the resolved graph
            self._createGraph(lookAt)
            labelImages = {}
            for t in timesteps:
                # use image provider plugin to load labelimage
                labelImage = self._readLabelImage(int(t))
                labelImages[t] = labelImage

            # compute new object features
            objectFeatures = self._computeObjectFeatures(labelImages)

            # load transition classifier if any
            if transition_classifier_filename is not None:
                getLogger().info("\tLoading transition classifier")
                transitionClassifier = probabilitygenerator.RandomForestClassifier(
                    transition_classifier_path, transition_classifier_filename)
            else:
                getLogger().info("\tUsing distance based transition energies")
                transitionClassifier = None

            # run min-cost max-flow to find gaps worthy to be closed
            getLogger().info("Running close gaps on a thresholding based method")

            ctcArcFlowMap = self._thresholdGapClosing(objectFeatures, transitionClassifier, threshold)

            # fuse results into a new solution
            self.result = self._saveTracks(ctcArcFlowMap, input_ctc, output_ctc)
