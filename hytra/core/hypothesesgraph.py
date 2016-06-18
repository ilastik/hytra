import networkx as nx
import numpy as np
from sklearn.neighbors import KDTree
import hytra.core.jsongraph
from hytra.core.jsongraph import negLog, listify
from hytra.core.progressbar import ProgressBar

def getTraxelFeatureVector(traxel, featureName, maxNumDimensions=3):
    """
    extract a feature vector from a traxel
    """
    result = []
    for i in range(maxNumDimensions):
        try:
            result.append(traxel.get_feature_value(featureName, i))
        except:
            if i == 0:
                print("Error when accessing feature {}[{}] for traxel (Id={},Timestep={})".format(featureName,
                                                                                                  i,
                                                                                                  traxel.Id,
                                                                                                  traxel.Timestep))
                print "Available features are: "
                print traxel.print_available_features()
                raise Exception
            else:
                print("Error: Classifier was trained with less merger than maxNumObjects {}.".format(maxNumDimensions))
                raise Exception
    return result


class NodeMap:
    """
    To access per node features of the hypotheses graph, 
    this node map provides the same interface as pgmlink's NodeMaps
    """

    def __init__(self, graph, attributeName):
        self.__graph = graph
        self.__attributeName = attributeName

    def __getitem__(self, key):
        return self.__graph.node[key][self.__attributeName]


class HypothesesGraph:
    """
    Replacement for pgmlink's hypotheses graph, 
    with a similar API so it can be used as drop-in replacement.

    Internally it uses [networkx](http://networkx.github.io/) to construct the graph.
    """

    def __init__(self):
        self._graph = nx.DiGraph()

    def nodeIterator(self):
        return self._graph.nodes_iter()

    def arcIterator(self):
        return self._graph.edges_iter()

    def countNodes(self):
        return self._graph.number_of_nodes()

    def countArcs(self):
        return self._graph.number_of_edges()

    @staticmethod
    def source(edge):
        return edge[0]

    @staticmethod
    def target(edge):
        return edge[1]

    def _findNearestNeighbors(self, kdtreeObjectPair, traxel, numNeighbors, maxNeighborDist):
        """
        Return a list of object IDs which are the 'numNeighbors' closest elements 
        in the kdtree less than maxNeighborDist away of the traxel.
        """
        kdtree, objectIdList = kdtreeObjectPair
        if len(objectIdList) <= numNeighbors:
            return objectIdList
        distances, neighbors = kdtree.query(self._extractCenter(traxel), k=numNeighbors, return_distance=True)
        return [objectIdList[index] for distance, index in zip(distances[0], neighbors[0]) if
                distance < maxNeighborDist]

    def _extractCenter(self, traxel):
        try:
            # python traxelstore
            if 'com' in traxel.Features:
                return traxel.Features['com']
            else:
                return traxel.Features['RegionCenter']
        except:
            # C++ pgmlink traxelstore
            try:
                return getTraxelFeatureVector(traxel, 'com')
            except:
                try:
                    return getTraxelFeatureVector(traxel, 'RegionCenter')
                except:
                    raise ValueError('given traxel (t={},id={}) does not have \
                        "com" or "RegionCenter"'.format(traxel.Timestep, traxel.Id))

    def _traxelMightDivide(self, traxel, divisionThreshold):
        assert 'divProb' in traxel.Features
        return traxel.Features['divProb'][0] > divisionThreshold

    def _buildFrameKdTree(self, traxelDict):
        """
        Collect the centers of all traxels and their ids of this frame's traxels.
        Then build a kdtree and return (kdtree, listOfObjectIdsInFrame), where the second argument
        is needed to decode the object id of the nearest neighbors in _findNearestNeighbors().
        """
        objectIdList = []
        features = []
        for obj, traxel in traxelDict.iteritems():
            if obj == 0:
                continue
            objectIdList.append(obj)
            features.append(list(self._extractCenter(traxel)))

        return (KDTree(features, metric='euclidean'), objectIdList)

    def _addNodesForFrame(self, frame, traxelDict):
        """
        Insert nodes for all objects in this frame, with the attribute "traxel"
        """
        for obj, traxel in traxelDict.iteritems():
            if obj == 0:
                continue
            self._graph.add_node((frame, obj), traxel=traxel)

    def buildFromTraxelstore(self, traxelstore, maxNeighborDist=200, numNearestNeighbors=1,
                             forwardBackwardCheck=True, withDivisions=True, divisionThreshold=0.1):
        """
        Takes a python traxelstore containing traxel features and finds probable links between frames.
        """
        assert (traxelstore is not None)
        assert (len(traxelstore.TraxelsPerFrame) > 0)

        def checkNodeWhileAddingLinks(frame, obj):
            if not (frame, obj) in self._graph:
                print("Adding node ({}, {}) when setting up links".format(frame, obj))

        kdTreeNextFrame = None
        for frame in range(len(traxelstore.TraxelsPerFrame.keys()) - 1):
            if frame > 0:
                kdTreeThisFrame = kdTreeNextFrame
            else:
                kdTreeThisFrame = self._buildFrameKdTree(traxelstore.TraxelsPerFrame[frame])
                self._addNodesForFrame(frame, traxelstore.TraxelsPerFrame[frame])

            kdTreeNextFrame = self._buildFrameKdTree(traxelstore.TraxelsPerFrame[frame + 1])
            self._addNodesForFrame(frame + 1, traxelstore.TraxelsPerFrame[frame + 1])

            # find forward links
            for obj, traxel in traxelstore.TraxelsPerFrame[frame].iteritems():
                divisionPreservingNumNearestNeighbors = numNearestNeighbors
                if divisionPreservingNumNearestNeighbors < 2 \
                        and withDivisions \
                        and self._traxelMightDivide(traxel, divisionThreshold):
                    divisionPreservingNumNearestNeighbors = 2
                neighbors = self._findNearestNeighbors(kdTreeNextFrame,
                                                       traxel,
                                                       divisionPreservingNumNearestNeighbors,
                                                       maxNeighborDist)
                for n in neighbors:
                    checkNodeWhileAddingLinks(frame, obj)
                    checkNodeWhileAddingLinks(frame + 1, n)
                    self._graph.add_edge((frame, obj), (frame + 1, n))

            # find backward links
            if forwardBackwardCheck:
                for obj, traxel in traxelstore.TraxelsPerFrame[frame + 1].iteritems():
                    neighbors = self._findNearestNeighbors(kdTreeThisFrame,
                                                           traxel,
                                                           numNearestNeighbors,
                                                           maxNeighborDist)
                    for n in neighbors:
                        checkNodeWhileAddingLinks(frame, n)
                        checkNodeWhileAddingLinks(frame + 1, obj)
                        self._graph.add_edge((frame, n), (frame + 1, obj))

    def generateTrackletGraph(self):
        raise NotImplementedError()

    def getNodeTraxelMap(self):
        return NodeMap(self._graph, 'traxel')

    def getNodeTrackletMap(self):
        raise NotImplementedError()


def convertHypothesesGraphToJsonGraph(hypothesesGraph,
                                      nodeIterator,
                                      arcIterator,
                                      withTracklets,
                                      maxNumObjects,
                                      numElements,
                                      traxelMap,
                                      detectionProbabilityFunc,
                                      transitionProbabilityFunc,
                                      boundaryCostMultiplierFunc,
                                      divisionProbabilityFunc):
    '''
    Build a json representation of this hypotheses graph, by transforming the probabilities for certain
    events (given by the `*ProbabilityFunc`-functions per traxel) into energies. If the given graph
    contained tracklets (`withTracklets`), then also the probabilities over all contained traxels will be
    accumulated for those nodes in the graph.

    The `hypothesesGraph` as well as `nodeIterator` and `arcIterator` are needed as parameters to 
    support the legacy pgmlink-style hypotheses graph as well.

    ** Parameters: **

    * `hypothesesGraph`: graph whose nodes and edges we are about to traverse.
    * `nodeIterator`: node iterator
    * `arcIterator`: arc iterator
    * `withTracklets`: whether tracklets are used
    * `maxNumObjects`: the max number of objects per detections
    * `numElements`: number of nodes + number of edges (for progress bar)
    * `traxelMap`: mapping from graph-node to list of traxels (in a tracklet)
    * `detectionProbabilityFunc`: should take a traxel and return its detection probabilities
     ([prob0objects, prob1object,...])
    * `transitionProbabilityFunc`: should take two traxels and return this link's probabilities
     ([prob0objectsInTransition, prob1objectsInTransition,...])
    * `boundaryCostMultiplierFunc`: should take a traxel and return a scalar multiplier between 0 and 1 for the
     appearance/disappearance cost that depends on the traxel's distance to the spacial and time boundary
    * `divisionProbabilityFunc`: should take a traxel and return its division probabilities
     ([probNoDiv, probDiv])
    '''

    progressBar = ProgressBar(stop=numElements)
    trackingGraph = hytra.core.jsongraph.JsonTrackingGraph()

    # add all detections to JSON
    for n in nodeIterator:
        if not withTracklets:
            # only one traxel, but make it a list so everything below works the same
            traxels = [traxelMap[n]]
        else:
            traxels = traxelMap[n]

        # accumulate features over all contained traxels
        previousTraxel = None
        detectionFeatures = np.zeros(maxNumObjects + 1)
        for t in traxels:
            detectionFeatures += np.array(negLog(detectionProbabilityFunc(t)))
            if previousTraxel is not None:
                detectionFeatures += np.array(negLog(transitionProbabilityFunc(previousTraxel, t)))
            previousTraxel = t

        detectionFeatures = listify(list(detectionFeatures))

        # division only if probability is big enough
        divisionFeatures = divisionProbabilityFunc(traxels[-1])
        if divisionFeatures is not None:
            divisionFeatures = listify(negLog(divisionFeatures))

        # appearance/disappearance
        appearanceFeatures = listify([0.0] + [boundaryCostMultiplierFunc(traxels[0])] * maxNumObjects)
        disappearanceFeatures = listify([0.0] + [boundaryCostMultiplierFunc(traxels[-1])] * maxNumObjects)

        trackingGraph.addDetectionHypothesesFromTracklet(traxels,
                                                         detectionFeatures,
                                                         divisionFeatures,
                                                         appearanceFeatures,
                                                         disappearanceFeatures,
                                                         timestep=[traxels[0].Timestep, traxels[-1].Timestep])
        progressBar.show()

    # add all links
    for a in arcIterator:
        if not withTracklets:
            srcTraxel = traxelMap[hypothesesGraph.source(a)]
            destTraxel = traxelMap[hypothesesGraph.target(a)]
        else:
            srcTraxel = traxelMap[hypothesesGraph.source(a)][-1]  # src is last of the traxels in source tracklet
            destTraxel = traxelMap[hypothesesGraph.target(a)][0]  # dest is first of traxels in destination tracklet
        src = trackingGraph.traxelIdPerTimestepToUniqueIdMap[str(srcTraxel.Timestep)][str(srcTraxel.Id)]
        dest = trackingGraph.traxelIdPerTimestepToUniqueIdMap[str(destTraxel.Timestep)][str(destTraxel.Id)]

        features = listify(negLog(transitionProbabilityFunc(srcTraxel, destTraxel)))
        trackingGraph.addLinkingHypotheses(src, dest, features)
        progressBar.show()

    return trackingGraph
