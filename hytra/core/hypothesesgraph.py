from __future__ import print_function, absolute_import, nested_scopes, generators, division, with_statement, unicode_literals
import logging
import copy
import networkx as nx
import numpy as np
from sklearn.neighbors import KDTree
import hytra.core.jsongraph
from hytra.core.jsongraph import negLog, listify
from hytra.util.progressbar import DefaultProgressVisitor


def getLogger():
    ''' logger to be used in this module '''
    return logging.getLogger(__name__)


def getTraxelFeatureVector(traxel, featureName, maxNumDimensions=3):
    """
    extract a feature vector from a traxel
    """
    result = []
    for i in range(maxNumDimensions):
        try:
            result.append(traxel.get_feature_value(str(featureName), i))
        except:
            if i == 0:
                getLogger().error("Error when accessing feature {}[{}] for traxel (Id={},Timestep={})".format(featureName,
                                                                                                              i,
                                                                                                              traxel.Id,
                                                                                                              traxel.Timestep))
                getLogger().error("Available features are: ")
                getLogger().error(traxel.print_available_features())
                raise Exception
            else:
                getLogger().error("Error: Classifier was trained with less merger than maxNumObjects {}.".format(maxNumDimensions))
                raise Exception
    return result


class NodeMap(object):
    """
    To access per node features of the hypotheses graph,
    this node map provides the same interface as pgmlink's NodeMaps
    """

    def __init__(self, graph, attributeName):
        self.__graph = graph
        self.__attributeName = attributeName

    def __getitem__(self, key):
        return self.__graph.node[key][self.__attributeName]


class HypothesesGraph(object):
    """
    Replacement for pgmlink's hypotheses graph,
    with a similar API so it can be used as drop-in replacement.

    Internally it uses [networkx](http://networkx.github.io/) to construct the graph.

    Use the insertEnergies() method to populate the nodes and arcs with the energies for different
    configurations (according to DPCT's JSON style'), derived from given probability generation functions.

    **Notes:** `self._graph.node`'s are indexed by tuples (int(timestep), int(id)), and contain either a
    single `'traxel'` attribute, or a list of traxels in `'tracklet'`.
    Nodes also get a unique ID assigned once they are added to the graph.
    """

    def __init__(self):
        self._graph = nx.DiGraph()
        self.withTracklets = False
        self.allowLengthOneTracks = True
        self._nextNodeUuid = 0
        self.progressVisitor=DefaultProgressVisitor()

    def nodeIterator(self):
        return self._graph.nodes_iter()

    def arcIterator(self):
        return self._graph.edges_iter()

    def countNodes(self):
        return self._graph.number_of_nodes()

    def countArcs(self):
        return self._graph.number_of_edges()
    
    def hasNode(self, node):
        return self._graph.has_node(node)
    
    def hasEdge(self, u, v):
        return self._graph.has_edge(u, v)

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
        distances, neighbors = kdtree.query([self._extractCenter(
            traxel)], k=numNeighbors, return_distance=True)
        return [objectIdList[index] for distance, index in zip(distances[0], neighbors[0]) if
                distance < maxNeighborDist]

    def _extractCenter(self, traxel):
        try:
            # python probabilityGenerator
            if 'com' in traxel.Features:
                return traxel.Features['com']
            else:
                return traxel.Features['RegionCenter']
        except:
            # C++ pgmlink probabilityGenerator
            try:
                return getTraxelFeatureVector(traxel, 'com')
            except:
                try:
                    return getTraxelFeatureVector(traxel, 'RegionCenter')
                except:
                    raise ValueError('given traxel (t={},id={}) does not have '
                                     '"com" or "RegionCenter"'.format(traxel.Timestep, traxel.Id))

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
            self._graph.add_node((frame, obj), traxel=traxel, id=self._nextNodeUuid)
            self._nextNodeUuid += 1
    
    def addNodeFromTraxel(self, traxel, **kwargs):
        """
        Insert a single node specified by a traxel.
        All keyword arguments are passed to the node as well.
        """
        assert(traxel is not None)
        assert(not self.withTracklets)
        self._graph.add_node((traxel.Timestep, traxel.Id), traxel=traxel, id=self._nextNodeUuid, **kwargs)
        self._nextNodeUuid += 1

    def buildFromProbabilityGenerator(self, probabilityGenerator, maxNeighborDist=200, numNearestNeighbors=1,
                                      forwardBackwardCheck=True, withDivisions=True, divisionThreshold=0.1, skipLinks=1):
        """
        Takes a python probabilityGenerator containing traxel features and finds probable links between frames.
        Builds a kdTree with the 'numNearestneighbors' for each frame and adds the nodes. In the same iteration, it adds
        a number of 'skipLinks' between the nodes separated by 'skipLinks' frames.
        """
        assert (probabilityGenerator is not None)
        assert (len(probabilityGenerator.TraxelsPerFrame) > 0)
        assert (skipLinks > 0)

        def checkNodeWhileAddingLinks(frame, obj):
            if (frame, obj) not in self._graph:
                getLogger().warning("Adding node ({}, {}) when setting up links".format(frame, obj))

        kdTreeFrames = [None]*(skipLinks + 1)
        # len(probabilityGenerator.TraxelsPerFrame.keys()) is NOT an indicator for the total number of frames,
        # because an empty frame does not create a key in the dictionary. E.g. for one frame in the middle of the
        # dataset, we won't access the last one.
        # Idea: take the max key in the dict. Remember, frame numbering starts with 0.
        frameMax = max(probabilityGenerator.TraxelsPerFrame.keys())
        frameMin = min(probabilityGenerator.TraxelsPerFrame.keys())
        numFrames = frameMax - frameMin + 1

        self.progressVisitor.showState("Probability Generator")

        countFrames = 0
        for frame in range(numFrames):
            countFrames += 1
            self.progressVisitor.showProgress(countFrames/float(numFrames))
            if frame > 0:
                del kdTreeFrames[0] # this is the current frame
                if frame + skipLinks < numFrames and frameMin + frame + skipLinks in probabilityGenerator.TraxelsPerFrame.keys():
                    kdTreeFrames.append(self._buildFrameKdTree(probabilityGenerator.TraxelsPerFrame[frameMin + frame + skipLinks]))
                    self._addNodesForFrame(frameMin + frame + skipLinks, probabilityGenerator.TraxelsPerFrame[frameMin + frame + skipLinks])
            else:
                for i in range(0, skipLinks+1):
                    if frameMin + frame + i in probabilityGenerator.TraxelsPerFrame.keys(): # empty frame
                        kdTreeFrames[i] = self._buildFrameKdTree(probabilityGenerator.TraxelsPerFrame[frameMin + frame + i])
                        self._addNodesForFrame(frameMin + frame + i, probabilityGenerator.TraxelsPerFrame[frameMin + frame + i])

            # find forward links
            if frameMin + frame in probabilityGenerator.TraxelsPerFrame.keys(): # 'frame' could be empty
                for obj, traxel in probabilityGenerator.TraxelsPerFrame[frameMin + frame].iteritems():
                    divisionPreservingNumNearestNeighbors = numNearestNeighbors
                    if divisionPreservingNumNearestNeighbors < 2 \
                            and withDivisions \
                            and self._traxelMightDivide(traxel, divisionThreshold):
                        divisionPreservingNumNearestNeighbors = 2
                    for i in range(1, skipLinks+1):
                        if frame + i < numFrames and frameMin + frame + i in probabilityGenerator.TraxelsPerFrame.keys():
                            neighbors = (self._findNearestNeighbors(kdTreeFrames[i],
                                                               traxel,
                                                               divisionPreservingNumNearestNeighbors,
                                                               maxNeighborDist))
                            # type(neighbors) is list
                            for n in neighbors:
                                checkNodeWhileAddingLinks(frameMin + frame, obj)
                                checkNodeWhileAddingLinks(frameMin + frame + i, n)
                                self._graph.add_edge((frameMin + frame, obj), (frameMin + frame + i, n))
                                self._graph.edge[frameMin + frame, obj][frameMin + frame + i, n]['src'] = self._graph.node[(frameMin + frame, obj)]['id']
                                self._graph.edge[frameMin + frame, obj][frameMin + frame + i, n]['dest'] = self._graph.node[(frameMin + frame + i, n)]['id']

            # find backward links
            if forwardBackwardCheck:
                for i in range(1, skipLinks+1):
                    if frame + i < numFrames:
                        if frameMin + frame + i in probabilityGenerator.TraxelsPerFrame.keys(): # empty frame
                            for obj, traxel in probabilityGenerator.TraxelsPerFrame[frameMin + frame + i].iteritems():
                                if kdTreeFrames[0] is not None:
                                    neighbors = (self._findNearestNeighbors(kdTreeFrames[0],
                                                                       traxel,
                                                                       numNearestNeighbors,
                                                                       maxNeighborDist))
                                    for n in neighbors:
                                        checkNodeWhileAddingLinks(frameMin + frame, n)
                                        checkNodeWhileAddingLinks(frameMin + frame + i, obj)
                                        self._graph.add_edge((frameMin + frame, n), (frameMin + frame + i, obj))
                                        self._graph.edge[frameMin + frame, n][frameMin + frame + i, obj]['src'] = self._graph.node[(frameMin + frame, n)]['id']
                                        self._graph.edge[frameMin + frame, n][frameMin + frame + i, obj]['dest'] = self._graph.node[(frameMin + frame + i, obj)]['id']

    def generateTrackletGraph(self):
        '''
        **Return** a new hypotheses graph where chains of detections with only one possible 
        incoming/outgoing transition are contracted into one node in the graph.
        The returned graph will have `withTracklets` set to `True`!

        The `'tracklet'` node map contains a list of traxels that each node represents.
        '''
        getLogger().info("generating tracklet graph...")
        tracklet_graph = copy.copy(self)
        tracklet_graph._graph = tracklet_graph._graph.copy()
        tracklet_graph.withTracklets = True
        tracklet_graph.referenceTraxelGraph = self
        tracklet_graph.progressVisitor = self.progressVisitor

        self.progressVisitor.showState("Initializing Tracklet Graph")
        # initialize tracklet map to contain a list of only one traxel per node
        countNodes = 0
        numNodes = tracklet_graph.countNodes()
        for node in tracklet_graph._graph.nodes_iter():
            countNodes += 1
            self.progressVisitor.showProgress(countNodes/float(numNodes))
            tracklet_graph._graph.node[node]['tracklet'] = [tracklet_graph._graph.node[node]['traxel']]
            del tracklet_graph._graph.node[node]['traxel']

        # set up a list of links that indicates whether the target's in- and source's out-degree
        # are one, meaning the edge can be contracted
        links_to_be_contracted = []
        node_remapping = {}
        self.progressVisitor.showState("Finding Tracklets in Graph")
        countEdges = 0
        numEdges = tracklet_graph.countArcs()
        for edge in tracklet_graph._graph.edges_iter():
            countEdges += 1
            self.progressVisitor.showProgress(countEdges/float(numEdges))
            if tracklet_graph._graph.out_degree(edge[0]) == 1 and tracklet_graph._graph.in_degree(edge[1]) == 1:
                links_to_be_contracted.append(edge)
                for i in [0, 1]:
                    node_remapping[edge[i]] = edge[i]

        # apply edge contraction
        self.progressVisitor.showState("Contracting Edges in Tracklet Graph")
        countLinks = 0
        numLinks = len(links_to_be_contracted)
        for edge in links_to_be_contracted:
            countLinks += 1
            self.progressVisitor.showProgress(countLinks/float(numLinks))
            src = node_remapping[edge[0]]
            dest = node_remapping[edge[1]]
            if tracklet_graph._graph.in_degree(src) == 0 and tracklet_graph._graph.out_degree(dest) == 0:
                # if this tracklet would contract to a single node without incoming or outgoing edges,
                # then do NOT contract, as our tracking cannot handle length-one-tracks
                continue
            
            tracklet_graph._graph.node[src]['tracklet'].extend(tracklet_graph._graph.node[dest]['tracklet'])
            # duplicate out arcs with new source
            for out_edge in tracklet_graph._graph.out_edges(dest):
                tracklet_graph._graph.add_edge(src, out_edge[1])
            # adjust node remapping to point to new source for all contracted traxels
            for t in tracklet_graph._graph.node[dest]['tracklet']:
                node_remapping[(t.Timestep, t.Id)] = src
            tracklet_graph._graph.remove_node(dest)

        getLogger().info("tracklet graph has {} nodes and {} edges (before {},{})".format(
            tracklet_graph.countNodes(), tracklet_graph.countArcs(), self.countNodes(), self.countArcs()))

        return tracklet_graph

    def getNodeTraxelMap(self):
        return NodeMap(self._graph, 'traxel')

    def getNodeTrackletMap(self):
        return NodeMap(self._graph, 'tracklet')

    def insertEnergies(self,
                       maxNumObjects,
                       detectionProbabilityFunc,
                       transitionProbabilityFunc,
                       boundaryCostMultiplierFunc,
                       divisionProbabilityFunc,
                       skipLinksBias):
        '''
        Insert energies for detections, divisions and links into the hypotheses graph, 
        by transforming the probabilities for certain
        events (given by the `*ProbabilityFunc`-functions per traxel) into energies. If the given graph
        contained tracklets (`self.withTracklets is True`), then also the probabilities over all contained traxels will be
        accumulated for those nodes in the graph.

        The energies are stored in the networkx graph under the following attribute names (to match the format for solvers):
        * detection energies: `self._graph.node[n]['features']`
        * division energies: `self._graph.node[n]['divisionFeatures']`
        * appearance energies: `self._graph.node[n]['appearanceFeatures']`
        * disappearance energies: `self._graph.node[n]['disappearanceFeatures']`
        * transition energies: `self._graph.edge[src][dest]['features']`
        * additionally we also store the timestep (range for traxels) per node as `timestep` attribute

        ** Parameters: **

        * `maxNumObjects`: the max number of objects per detections
        * `detectionProbabilityFunc`: should take a traxel and return its detection probabilities
         ([prob0objects, prob1object,...])
        * `transitionProbabilityFunc`: should take two traxels and return this link's probabilities
         ([prob0objectsInTransition, prob1objectsInTransition,...])
        * `boundaryCostMultiplierFunc`: should take a traxel and a boolean that is true if we are seeking for an appearance cost multiplier, 
         false for disappearance, and return a scalar multiplier between 0 and 1 for the
         appearance/disappearance cost that depends on the traxel's distance to the spacial and time boundary
        * `divisionProbabilityFunc`: should take a traxel and return its division probabilities ([probNoDiv, probDiv])
        '''
        numElements = self._graph.number_of_nodes() + self._graph.number_of_edges()
        self.progressVisitor.showState("Inserting energies")

        # insert detection probabilities for all detections (and some also get a div probability)
        countElements = 0
        for n in self._graph.nodes_iter():
            countElements += 1
            if not self.withTracklets:
                # only one traxel, but make it a list so everything below works the same
                traxels = [self._graph.node[n]['traxel']]
            else:
                traxels = self._graph.node[n]['tracklet']

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
            appearanceFeatures = listify([0.0] + [boundaryCostMultiplierFunc(traxels[0], True)] * maxNumObjects)
            disappearanceFeatures = listify([0.0] + [boundaryCostMultiplierFunc(traxels[-1], False)] * maxNumObjects)

            self._graph.node[n]['features'] = detectionFeatures
            if divisionFeatures is not None:
                self._graph.node[n]['divisionFeatures'] = divisionFeatures
            self._graph.node[n]['appearanceFeatures'] = appearanceFeatures
            self._graph.node[n]['disappearanceFeatures'] = disappearanceFeatures
            self._graph.node[n]['timestep'] = [traxels[0].Timestep, traxels[-1].Timestep]

            self.progressVisitor.showProgress(countElements/float(numElements))

        # insert transition probabilities for all links
        for a in self._graph.edges_iter():
            countElements += 1
            self.progressVisitor.showProgress(countElements/float(numElements))

            if not self.withTracklets:
                srcTraxel = self._graph.node[self.source(a)]['traxel']
                destTraxel = self._graph.node[self.target(a)]['traxel']
            else:
                srcTraxel = self._graph.node[self.source(a)]['tracklet'][-1]  # src is last of the traxels in source tracklet
                destTraxel = self._graph.node[self.target(a)]['tracklet'][0]  # dest is first of traxels in destination tracklet

            features = listify(negLog(transitionProbabilityFunc(srcTraxel, destTraxel)))

            # add feature for additional Frames. Since we do not want these edges to be primarily taken, we add a bias to the edge. Now: hard coded, future: parameter
            frame_gap = destTraxel.Timestep - srcTraxel.Timestep

            # 1. method
            if frame_gap > 1:
                features[1][0] = features[1][0] + skipLinksBias*frame_gap

            # # 2. method
            # # introduce a new energies like: [[6], [15]] -> [[6, 23], [15, 23]] for first links and
            # # [[6], [15]] -> [[23, 6], [23, 15]] for second links, and so on for 3rd order links
            # # !!! this will introduce a new weight in the weight.json file. For the 2nd link, comes in 2nd row and so on.
            # # drawback: did not manage to adjust parameter to get sensible results.
            # for feat in features:
            #     for i in range(frame_gap):
            #         feat.append(23)
            #     if frame_gap > 1:
            #         feat[frame_gap-1], feat[0] = feat[0], feat[frame_gap-1]


            self._graph.edge[a[0]][a[1]]['src'] = self._graph.node[a[0]]['id']
            self._graph.edge[a[0]][a[1]]['dest'] = self._graph.node[a[1]]['id']
            self._graph.edge[a[0]][a[1]]['features'] = features

    def getMappingsBetweenUUIDsAndTraxels(self):
        '''
        Extract the mapping from UUID to traxel and vice versa from the networkx graph.

        ** Returns: a tuple of **

        * `traxelIdPerTimestepToUniqueIdMap`: a dictionary of the structure `{str(timestep):{str(labelimageId):int(uuid), 
         str(labelimageId):int(uuid), ...}, str(nextTimestep):{}, ...}`
        * `uuidToTraxelMap`: a dictionary with keys = int(uuid), values = list(of timestep-Id-tuples (int(Timestep), int(Id)))
        '''

        uuidToTraxelMap = {}
        traxelIdPerTimestepToUniqueIdMap = {}

        for n in self._graph.nodes_iter():
            uuid = self._graph.node[n]['id']
            traxels = []
            if self.withTracklets:
                traxels = self._graph.node[n]['tracklet']
            else:
                traxels = [self._graph.node[n]['traxel']]
            uuidToTraxelMap[uuid] = [(t.Timestep, t.Id) for t in traxels]

            for t in uuidToTraxelMap[uuid]:
                traxelIdPerTimestepToUniqueIdMap.setdefault(str(t[0]), {})[str(t[1])] = uuid

        # sort the list of traxels per UUID by their timesteps
        for v in uuidToTraxelMap.values():
            v.sort(key=lambda timestepIdTuple: timestepIdTuple[0])

        return traxelIdPerTimestepToUniqueIdMap, uuidToTraxelMap

    def toTrackingGraph(self, noFeatures=False):
        '''
        Create a dictionary representation of this graph which can be passed to the solvers directly.
        The resulting graph (=model) is wrapped within a `hytra.jsongraph.JsonTrackingGraph` structure for convenience.
        If `noFeatures` is `True`, then only the structure of the graph will be exported.
        '''
        requiredNodeAttribs = ['id']
        requiredLinkAttribs = ['src', 'dest']

        if not noFeatures:
            requiredNodeAttribs.append('features')
            requiredLinkAttribs.append('features')

        def translateNodeToDict(n):
            result = {}
            attrs = self._graph.node[n]
            for k in ['id', 'features', 'appearanceFeatures', 'disappearanceFeatures', 'divisionFeatures', 'timestep']:
                if k in attrs:
                    result[k] = attrs[k]
                elif k in requiredNodeAttribs:
                    raise ValueError('Cannot use graph nodes without assigned ID and features, run insertEnergies() first')
            return result

        def translateLinkToDict(l):
            result = {}
            attrs = self._graph.edge[l[0]][l[1]]
            for k in ['src', 'dest', 'features']:
                if k in attrs:
                    result[k] = attrs[k]
                elif k in requiredLinkAttribs:
                    raise ValueError('Cannot use graph links without source, target, and features, run insertEnergies() first')
            return result

        traxelIdPerTimestepToUniqueIdMap, _ = self.getMappingsBetweenUUIDsAndTraxels()
        model = {
            'segmentationHypotheses':[translateNodeToDict(n) for n in self._graph.nodes_iter()],
            'linkingHypotheses':[translateLinkToDict(e) for e in self._graph.edges_iter()],
            'divisionHypotheses':[],
            'traxelToUniqueId':traxelIdPerTimestepToUniqueIdMap,
            'settings':{'statesShareWeights':True,
                        'allowPartialMergerAppearance':False,
                        'requireSeparateChildrenOfDivision':True,
                        'optimizerEpGap':0.01,
                        'optimizerVerbose':True,
                        'optimizerNumThreads':1
                       }
            }

        # extract exclusion sets:
        exclusions = set([])
        for n in self._graph.nodes_iter():
            if self.withTracklets:
                traxel = self._graph.node[n]['tracklet'][0]
            else:
                traxel = self._graph.node[n]['traxel']
            
            if traxel.conflictingTraxelIds is not None:
                if self.withTracklets:
                    getLogger().error("Exclusion constraints do not work with tracklets yet!")
                
                conflictingIds = [traxelIdPerTimestepToUniqueIdMap[str(traxel.Timestep)][str(i)] for i in traxel.conflictingTraxelIds]
                myId = traxelIdPerTimestepToUniqueIdMap[str(traxel.Timestep)][str(traxel.Id)]
                for ci in conflictingIds:
                    # insert pairwise exclusion constraints only, and always put the lower id first
                    if ci < myId:
                        exclusions.add((ci, myId))
                    else:
                        exclusions.add((myId, ci))

        model['exclusions'] = [list(t) for t in exclusions]

        # TODO: this recomputes the uuidToTraxelMap even though we have it already...
        trackingGraph = hytra.core.jsongraph.JsonTrackingGraph(
            model=model,
            progressVisitor=self.progressVisitor
        )
        return trackingGraph

    def insertSolution(self, resultDictionary):
        '''
        Add solution values to nodes and arcs from dictionary representation of solution.
        The resulting graph (=model) gets an additional property "value" that represents the number of objects inside a detection/arc
        Additionally a division indicator is saved in the node property "divisionValue".
        The link also gets a new attribute: the gap that is covered. E.g. 1, if consecutive timeframes, 2 if link skipping one timeframe.
        '''
        _, uuidToTraxelMap = self.getMappingsBetweenUUIDsAndTraxels()

        if self.withTracklets:
            traxelgraph = self.referenceTraxelGraph
        else:
            traxelgraph = self

        # reset all values
        for n in traxelgraph._graph.nodes_iter():
            traxelgraph._graph.node[n]['value'] = 0
            traxelgraph._graph.node[n]['divisionValue'] = False

        for e in traxelgraph._graph.edges_iter():
            traxelgraph._graph.edge[e[0]][e[1]]['value'] = 0

        # store values from dict
        for detection in resultDictionary["detectionResults"]:
            traxels = uuidToTraxelMap[detection["id"]]
            for traxel in traxels:
                traxelgraph._graph.node[traxel]['value'] = detection["value"]
            for internal_edge in zip(traxels,traxels[1:]):
                traxelgraph._graph.edge[internal_edge[0]][internal_edge[1]]['value'] = detection["value"]

        if "linkingResults" in resultDictionary and resultDictionary["linkingResults"] is not None: 
            for link in resultDictionary["linkingResults"]:
                source, dest = uuidToTraxelMap[link["src"]][-1], uuidToTraxelMap[link["dest"]][0]
                traxelgraph._graph.edge[source][dest]['value'] = link["value"]
                traxelgraph._graph.edge[source][dest]['gap'] = dest[0] - source[0]

        if "divisionResults" in resultDictionary and resultDictionary["divisionResults"] is not None:
            for division in resultDictionary["divisionResults"]:
                traxelgraph._graph.node[uuidToTraxelMap[division["id"]][-1]]['divisionValue'] = division["value"]

    def getSolutionDictionary(self):
        '''
        Return the solution encoded in the `value` and `divisionValue` attributes of nodes and edges
        as a python dictionary in the style that can be saved to JSON or sent to our solvers as ground truths.
        '''
        resultDictionary = {}

        if self.withTracklets:
            traxelgraph = self.referenceTraxelGraph
        else:
            traxelgraph = self

        detectionList = []
        divisionList = []
        linkList = []

        def checkAttributeValue(element, attribName, default):
            if attribName in element:
                return element[attribName]
            else:
                return default

        for n in traxelgraph._graph.nodes_iter():
            newDetection = {}
            newDetection['id'] = traxelgraph._graph.node[n]['id']
            newDetection['value'] = checkAttributeValue(traxelgraph._graph.node[n], 'value', 0)
            detectionList.append(newDetection)
            if 'divisionValue' in traxelgraph._graph.node[n]:
                newDivsion = {}
                newDivsion['id'] = traxelgraph._graph.node[n]['id']
                newDivsion['value'] = checkAttributeValue(traxelgraph._graph.node[n], 'divisionValue', False)
                divisionList.append(newDivsion)
                
        for a in traxelgraph.arcIterator():
            newLink = {}
            src = self.source(a)
            dest = self.target(a)
            newLink['src'] = traxelgraph._graph.node[src]['id']
            newLink['dest'] = traxelgraph._graph.node[dest]['id']
            newLink['value'] = checkAttributeValue(traxelgraph._graph.edge[src][dest], 'value', 0)
            newLink['gap'] = checkAttributeValue(traxelgraph._graph.edge[src][dest], 'gap', 1)

            linkList.append(newLink)

        resultDictionary["detectionResults"] = detectionList
        resultDictionary["linkingResults"] = linkList
        resultDictionary["divisionResults"] = divisionList

        return resultDictionary

    def countIncomingObjects(self, node):
        '''
        Once a solution was written to the graph, this returns the number of
        incoming objects of a node, and the number of active incoming edges.
        If the latter is greater than 1, this shows that we have a merger.
        '''
        numberOfIncomingObject = 0
        numberOfIncomingEdges = 0
        for in_edge in self._graph.in_edges(node):
            if 'value' in self._graph.edge[in_edge[0]][node]:
                numberOfIncomingObject += self._graph.edge[in_edge[0]][node]['value']
                numberOfIncomingEdges += 1
        return numberOfIncomingObject, numberOfIncomingEdges

    def countOutgoingObjects(self, node):
        '''
        Once a solution was written to the graph, this returns the number of
        outgoing objects of a node, and the number of active outgoing edges.
        If the latter is greater than 1, this shows that we have a merger splitting up, or a division.
        '''
        numberOfOutgoingObject = 0
        numberOfOutgoingEdges = 0
        for out_edge in self._graph.out_edges(node):
            if 'value' in self._graph.edge[node][out_edge[1]] and self._graph.edge[node][out_edge[1]]['value'] > 0:
                numberOfOutgoingObject += self._graph.edge[node][out_edge[1]]['value']
                numberOfOutgoingEdges += 1
        return numberOfOutgoingObject, numberOfOutgoingEdges

    def computeLineage(self, firstTrackId=2, firstLineageId=2, skipLinks=1):
        """
        computes lineage and track id for every node in the graph
        """

        update_queue = []
        # start lineages / tracks at 2, because 0 means background=black, 1 means misdetection in ilastik
        max_lineage_id = firstLineageId
        max_track_id = firstTrackId

        if self.withTracklets:
            traxelgraph = self.referenceTraxelGraph
        else:
            traxelgraph = self

        self.progressVisitor.showState("Compute lineage")

        # find start of lineages
        numElements = 2*traxelgraph.countNodes()
        countElements = 0
        for n in traxelgraph.nodeIterator():
            countElements += 1
            self.progressVisitor.showProgress(countElements/float(numElements))

            if traxelgraph.countIncomingObjects(n)[0] == 0 \
                and 'value' in traxelgraph._graph.node[n] \
                and traxelgraph._graph.node[n]['value'] > 0 \
                and (self.allowLengthOneTracks or traxelgraph.countOutgoingObjects(n)[0] > 0):
                # found start of a track
                update_queue.append((n,max_lineage_id,max_track_id))
                max_lineage_id += 1
                max_track_id   += 1
            else:
                traxelgraph._graph.node[n]["lineageId"] = None
                traxelgraph._graph.node[n]["trackId"] = None


        while len(update_queue) > 0:
            countElements += 1
            current_node,lineage_id,track_id = update_queue.pop()
            self.progressVisitor.showProgress(countElements/float(numElements))

            # if we did not run merger resolving, it can happen that we reach a node several times,
            # and would propagate the new lineage+track IDs to all descendants again! We simply
            # stop propagating in that case and just use the lineageID that reached the node first.
            if traxelgraph._graph.node[current_node].get("lineageId", None) is not None and \
                traxelgraph._graph.node[current_node].get("trackId", None) is not None:
                getLogger().debug("Several tracks are merging here, stopping a later one")
                continue

            # set a new trackID
            traxelgraph._graph.node[current_node]["lineageId"] = lineage_id
            traxelgraph._graph.node[current_node]["trackId"] = track_id

            numberOfOutgoingObject, numberOfOutgoingEdges = traxelgraph.countOutgoingObjects(current_node)
            
            if (numberOfOutgoingObject != numberOfOutgoingEdges):
                getLogger().warning("running lineage computation on unresolved graphs depends on a race condition")

            if 'divisionValue' in traxelgraph._graph.node[current_node] and traxelgraph._graph.node[current_node]['divisionValue']:
                assert(traxelgraph.countOutgoingObjects(current_node)[1] == 2)
                traxelgraph._graph.node[current_node]['children'] = []
                for a in traxelgraph._graph.out_edges(current_node):

                    if 'value' in traxelgraph._graph.edge[current_node][a[1]] and traxelgraph._graph.edge[current_node][a[1]]['value'] > 0:
                        traxelgraph._graph.node[a[1]]['gap'] = skipLinks
                        traxelgraph._graph.node[current_node]['children'].append(a[1])
                        traxelgraph._graph.node[a[1]]['parent'] = current_node
                        update_queue.append((traxelgraph.target(a),
                                            lineage_id,
                                            max_track_id))
                        max_track_id += 1
            else:
                if traxelgraph.countOutgoingObjects(current_node)[1] > 1:
                    getLogger().debug('Found merger splitting into several objects, propagating lineage and track to all descendants!')

                for a in traxelgraph._graph.out_edges(current_node):
                    if 'value' in traxelgraph._graph.edge[current_node][a[1]] and traxelgraph._graph.edge[current_node][a[1]]['value'] > 0:
                        if ('gap' in traxelgraph._graph.edge[current_node][a[1]] and traxelgraph._graph.edge[current_node][a[1]]['gap'] == 1) or 'gap' not in traxelgraph._graph.edge[current_node][a[1]]:
                            traxelgraph._graph.node[a[1]]['gap'] = 1
                            update_queue.append((traxelgraph.target(a),
                                            lineage_id,
                                            track_id))
                        if 'gap' in traxelgraph._graph.edge[current_node][a[1]] and traxelgraph._graph.edge[current_node][a[1]]['gap'] > 1:
                            traxelgraph._graph.node[a[1]]['gap'] = skipLinks
                            traxelgraph._graph.node[a[1]]['gap_parent'] = current_node
                            update_queue.append((traxelgraph.target(a),
                                            lineage_id,
                                            max_track_id))
                            max_track_id += 1

    def pruneGraphToSolution(self, distanceToSolution=0):
        '''
        creates a new pruned HypothesesGraph that around the result. Assumes that value==0 corresponds
        to unlabeled parts of the graph.
        distanceToSolution determines how many negative examples are included
        distanceToSolution = 0: only include negative edges that connect used objects
        distanceToSolution = 1: additionally include edges that connect used objects with unlabeled objects
        '''
        prunedGraph = HypothesesGraph()
        for n in self.nodeIterator():
            if 'value' in self._graph.node[n] and self._graph.node[n]['value'] > 0:
                prunedGraph._graph.add_node(n,**self._graph.node[n])

        for e in self.arcIterator():
            src = self.source(e)
            dest = self.target(e)
            if distanceToSolution == 0:
                if src in prunedGraph._graph and dest in prunedGraph._graph:
                    prunedGraph._graph.add_edge(src,dest,**self._graph.edge[src][dest])

        # TODO: can be optimized by looping over the pruned graph nodes(might sacrifice readability)
        for distance in range(1,distanceToSolution+1):
            for e in self.arcIterator():
                src = self.source(e)
                dest = self.target(e)
                if src in prunedGraph._graph or dest in prunedGraph._graph:
                    prunedGraph._graph.add_node(src,**self._graph.node[src])
                    prunedGraph._graph.add_node(dest,**self._graph.node[dest])
                    prunedGraph._graph.add_edge(src,dest,**self._graph.edge[src][dest])

        return prunedGraph
    
    def _getNodeAttribute(self, timestep, objectId, attribute):
        '''
        return some attribute of a certain node specified by timestep and objectId
        '''
        try:
            return self._graph.node[(int(timestep), int(objectId))][attribute]
        except KeyError:
            getLogger().error(attribute + ' not found in graph node properties, call computeLineage() first!')
            raise

    def getLineageId(self, timestep, objectId):
        '''
        return the lineage Id of a certain node specified by timestep and objectId
        '''
        if self.withTracklets:
            traxelgraph = self.referenceTraxelGraph
        else:
            traxelgraph = self
        return traxelgraph._getNodeAttribute(timestep, objectId, 'lineageId') 
    
    def getTrackId(self, timestep, objectId):
        '''
        return the track Id of a certain node specified by timestep and objectId
        '''
        if self.withTracklets:
            traxelgraph = self.referenceTraxelGraph
        else:
            traxelgraph = self
        return traxelgraph._getNodeAttribute(timestep, objectId, 'trackId')
