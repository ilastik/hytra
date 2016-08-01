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

"Anpassen aller DOOOOCS"

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
                # print node, neighb
                if node != neighb and node[0] == neighb[0] - 2 and node[2] == 'ending' and neighb[2] == 'starting':
                    addNode(node)
                    addNode(neighb)
                    self.Graph.add_edge((node[0], node[1]), (neighb[0], neighb[1]))
                    print node, neighb, "added edges"

        return self.Graph

    def _readLabelImage(self, timeframe):
        '''
        Should return the labelimage for the given timeframe
        '''
        raise NotImplementedError()

    def _minCostMaxFlowGapClosing(self, objectFeatures, transitionClassifier=None, transitionParameter=5.0):
        """
        Find the optimal assignments within the `Graph` by running min-cost max-flow from the
        `dpct` module.

        Converts the `Graph` to our JSON model structure, predicts the transition probabilities 
        either using the given transitionClassifier, or using distance-based probabilities.

        **returns** a `nodeFlowMap` and `arcFlowMap` holding information on the usage of the respective nodes and links

        **Note:** cannot use `networkx` flow methods because they don't work with floating point weights.
        """

        trackingGraph = JsonTrackingGraph()
        for node in self.Graph.nodes_iter():
            additionalFeatures = {}
            if len(self.Graph.in_edges(node)) == 0:
                additionalFeatures['appearanceFeatures'] = [[0], [0]]
            if len(self.Graph.out_edges(node)) == 0:
                additionalFeatures['disappearanceFeatures'] = [[0], [0]]
            uuid = trackingGraph.addDetectionHypotheses([[0], [1]], **additionalFeatures)
            self.Graph.node[node]['id'] = uuid

        for edge in self.Graph.edges_iter():
            src = self.Graph.node[edge[0]]['id']
            dest = self.Graph.node[edge[1]]['id']

            print src, self.Graph.node[edge[0]]
            print dest, self.Graph.node[edge[1]]

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

            trackingGraph.addLinkingHypotheses(src, dest, listify(negLog(probs)))

        # track
        import dpct
        weights = {"weights": [1, 1, 1, 1]}
        mergerResult = dpct.trackMaxFlow(trackingGraph.model, weights)

        # transform results to dictionaries that can be indexed by id or (src,dest)
        nodeFlowMap = dict([(int(d['id']), int(d['value'])) for d in mergerResult['detectionResults']])
        arcFlowMap = dict([((int(l['src']), int(l['dest'])), int(l['value'])) for l in mergerResult['linkingResults']])

        print nodeFlowMap, arcFlowMap
        return nodeFlowMap, arcFlowMap

    def _refineResult(self,
                      nodeFlowMap,
                      arcFlowMap,
                      traxelIdPerTimestepToUniqueIdMap):
        """
        Update the `self.result` dict by removing the mergers and adding the refined nodes and links.

        Operates on a `result` dictionary in our JSON result style with mergers,
        the resolved and unresolved graph as well as
        the `nodeFlowMap` and `arcFlowMap` obtained by running tracking on the `Graph`.

        Updates the `result` dictionary so that all merger nodes are removed but the new nodes
        are contained with the appropriate links and values.

        `mergerNodeFilter` and `mergerLinkFilter` are methods that can filter merger detections
        and links from the respective lists in the `result` dict.

        **Returns** the updated `result` dict, which is the same as the input `result` (works in-place)
        """

        # filter merger edges
        self.result['detectionResults'] = [r for r in self.result['detectionResults'] if mergerNodeFilter(r)]
        self.result['linkingResults'] = [r for r in self.result['linkingResults'] if mergerLinkFilter(r)]

        # add new nodes
        for node in self.Graph.nodes_iter():
            if self.Graph.node[node]['count'] > 1:
                newIds = self.Graph.node[node]['newIds']
                for newId in newIds:
                    uuid = traxelIdPerTimestepToUniqueIdMap[str(node[0])][str(newId)]
                    resolvedNode = (node[0], newId)
                    resolvedResultId = self.Graph.node[resolvedNode]['id']
                    newDetection = {'id': uuid, 'value': nodeFlowMap[resolvedResultId]}
                    self.result['detectionResults'].append(newDetection)

        # add new links
        for edge in self.Graph.edges_iter():
            newLink = {}
            newLink['src'] = traxelIdPerTimestepToUniqueIdMap[str(edge[0][0])][str(edge[0][1])]
            newLink['dest'] = traxelIdPerTimestepToUniqueIdMap[str(edge[1][0])][str(edge[1][1])]
            srcId = self.Graph.node[edge[0]]['id']
            destId = self.Graph.node[edge[1]]['id']
            newLink['value'] = arcFlowMap[(srcId, destId)]
            self.result['linkingResults'].append(newLink)

        return self.result

    def _computeObjectFeatures(self, labelImages):
        '''
        Return the features per object as nested dictionaries:
        { (int(Timestep), int(Id)):{ "FeatureName" : np.array(value), "NextFeature": ...} }
        '''
        pass

    # ------------------------------------------------------------
    def run(self, input_ctc, transition_classifier_filename=None, transition_classifier_path=None):
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
        print "LOOK AT", lookAt


        # ------------------------------------------------------------
        if lookAt == None:
            getLogger().info("There are no gaps to close, nothing to be done. Writing the output...")
            # segmentation
            labelImages = {}
            for t in timesteps:
                # use image provider plugin to load labelimage
                labelImage = self._readLabelImage(int(t))
                labelImages[t] = labelImage
        else:
            # set up unresolved graph and then refine the nodes to get the resolved graph
            self._createGraph(lookAt)
            labelImages = {}
            for t in timesteps:
                # use image provider plugin to load labelimage
                labelImage = self._readLabelImage(int(t))
                labelImages[t] = labelImage

            # ------------------------------------------------------------
            # compute new object features
            objectFeatures = self._computeObjectFeatures(labelImages)

            # ------------------------------------------------------------
            # load transition classifier if any
            if transition_classifier_filename is not None:
                getLogger().info("\tLoading transition classifier")
                transitionClassifier = probabilitygenerator.RandomForestClassifier(
                    transition_classifier_path, transition_classifier_filename)
            else:
                getLogger().info("\tUsing distance based transition energies")
                transitionClassifier = None

            # ------------------------------------------------------------
            # run min-cost max-flow to find gaps worthy to be closed
            getLogger().info("Running min-cost max-flow to find closed gaps assignments")

            nodeFlowMap, arcFlowMap = self._minCostMaxFlowGapClosing(objectFeatures, transitionClassifier)

            # ------------------------------------------------------------
            # fuse results into a new solution

            self.result = self._refineResult(nodeFlowMap,
                                             arcFlowMap,
                                             traxelIdPerTimestepToUniqueIdMap)
