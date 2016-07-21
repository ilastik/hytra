import logging
import copy
import networkx as nx
import numpy as np
from sklearn.neighbors import KDTree
import hytra.core.jsongraph
from hytra.core.jsongraph import negLog, listify
from hytra.util.progressbar import ProgressBar


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
            result.append(traxel.get_feature_value(featureName, i))
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
        self._nextNodeUuid = 0

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
        distances, neighbors = kdtree.query(self._extractCenter(
            traxel), k=numNeighbors, return_distance=True)
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

    def buildFromProbabilityGenerator(self, probabilityGenerator, maxNeighborDist=200, numNearestNeighbors=1,
                                      forwardBackwardCheck=True, withDivisions=True, divisionThreshold=0.1):
        """
        Takes a python traxelstore containing traxel features and finds probable links between frames.
        """
        assert (probabilityGenerator is not None)
        assert (len(probabilityGenerator.TraxelsPerFrame) > 0)

        def checkNodeWhileAddingLinks(frame, obj):
            if (frame, obj) not in self._graph:
                getLogger().warning("Adding node ({}, {}) when setting up links".format(frame, obj))

        kdTreeNextFrame = None
        for frame in range(len(probabilityGenerator.TraxelsPerFrame.keys()) - 1):
            if frame > 0:
                kdTreeThisFrame = kdTreeNextFrame
            else:
                kdTreeThisFrame = self._buildFrameKdTree(probabilityGenerator.TraxelsPerFrame[frame])
                self._addNodesForFrame(frame, probabilityGenerator.TraxelsPerFrame[frame])

            kdTreeNextFrame = self._buildFrameKdTree(probabilityGenerator.TraxelsPerFrame[frame + 1])
            self._addNodesForFrame(frame + 1, probabilityGenerator.TraxelsPerFrame[frame + 1])

            # find forward links
            for obj, traxel in probabilityGenerator.TraxelsPerFrame[frame].iteritems():
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
                for obj, traxel in probabilityGenerator.TraxelsPerFrame[frame + 1].iteritems():
                    neighbors = self._findNearestNeighbors(kdTreeThisFrame,
                                                           traxel,
                                                           numNearestNeighbors,
                                                           maxNeighborDist)
                    for n in neighbors:
                        checkNodeWhileAddingLinks(frame, n)
                        checkNodeWhileAddingLinks(frame + 1, obj)
                        self._graph.add_edge((frame, n), (frame + 1, obj))

    def generateTrackletGraph(self):
        '''
        **Return** a new hypotheses graph where chains of detections with only one possible 
        incoming/outgoing transition are contracted into one node in the graph.
        The returned graph will have `withTracklets` set to `True`!

        The `'tracklet'` node map contains a list of traxels that this node represents.
        '''
        getLogger().info("generating tracklet graph...")
        tracklet_graph = copy.copy(self)
        tracklet_graph._graph = tracklet_graph._graph.copy() 
        tracklet_graph.withTracklets = True

        # initialize tracklet map to contain a list of only one traxel per node
        for node in tracklet_graph._graph.nodes_iter():
            tracklet_graph._graph.node[node]['tracklet'] = [tracklet_graph._graph.node[node]['traxel']]
            del tracklet_graph._graph.node[node]['traxel']

        # set up a list of links that indicates whether the target's in- and source's out-degree
        # are one, meaning the edge can be contracted
        links_to_be_contracted = []
        node_remapping = {}
        for edge in tracklet_graph._graph.edges_iter():
            if tracklet_graph._graph.out_degree(edge[0]) == 1 and tracklet_graph._graph.in_degree(edge[1]) == 1:
                links_to_be_contracted.append(edge)
                for i in [0, 1]:
                    node_remapping[edge[i]] = edge[i]

        # apply edge contraction
        for edge in links_to_be_contracted:
            src = node_remapping[edge[0]]
            dest = node_remapping[edge[1]]
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
                       divisionProbabilityFunc):
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
        * `boundaryCostMultiplierFunc`: should take a traxel and return a scalar multiplier between 0 and 1 for the
         appearance/disappearance cost that depends on the traxel's distance to the spacial and time boundary
        * `divisionProbabilityFunc`: should take a traxel and return its division probabilities ([probNoDiv, probDiv])
        '''
        numElements = self._graph.number_of_nodes() + self._graph.number_of_edges() 
        progressBar = ProgressBar(stop=numElements)

        # insert detection probabilities for all detections (and some also get a div probability)
        for n in self._graph.nodes_iter():
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
            appearanceFeatures = listify([0.0] + [boundaryCostMultiplierFunc(traxels[0])] * maxNumObjects)
            disappearanceFeatures = listify([0.0] + [boundaryCostMultiplierFunc(traxels[-1])] * maxNumObjects)

            self._graph.node[n]['features'] = detectionFeatures
            if divisionFeatures is not None:
                self._graph.node[n]['divisionFeatures'] = divisionFeatures
            self._graph.node[n]['appearanceFeatures'] = appearanceFeatures
            self._graph.node[n]['disappearanceFeatures'] = disappearanceFeatures
            self._graph.node[n]['timestep'] = [traxels[0].Timestep, traxels[-1].Timestep]

            progressBar.show()

        # insert transition probabilities for all links
        for a in self._graph.edges_iter():
            if not self.withTracklets:
                srcTraxel = self._graph.node[self.source(a)]['traxel']
                destTraxel = self._graph.node[self.target(a)]['traxel']
            else:
                srcTraxel = self._graph.node[self.source(a)]['tracklet'][-1]  # src is last of the traxels in source tracklet
                destTraxel = self._graph.node[self.target(a)]['tracklet'][0]  # dest is first of traxels in destination tracklet

            features = listify(negLog(transitionProbabilityFunc(srcTraxel, destTraxel)))

            self._graph.edge[a[0]][a[1]]['src'] = self._graph.node[a[0]]['id']
            self._graph.edge[a[0]][a[1]]['dest'] = self._graph.node[a[1]]['id']
            self._graph.edge[a[0]][a[1]]['features'] = features
            
            progressBar.show()
        
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

    def toTrackingGraph(self):
        '''
        Create a dictionary representation of this graph which can be passed to the solvers directly.
        The resulting graph (=model) is wrapped within a `hytra.jsongraph.JsonTrackingGraph` structure for convenience.
        '''

        def translateNodeToDict(n):
            result = {}
            attrs = self._graph.node[n]
            for k in ['id', 'features', 'appearanceFeatures', 'disappearanceFeatures', 'divisionFeatures', 'timestep']:
                if k in attrs:
                    result[k] = attrs[k]
                elif k == 'features' or k == 'id':
                    raise ValueError('Cannot use graph nodes without assigned ID and features, run insertEnergies() first')
            return result
        
        def translateLinkToDict(l):
            result = {}
            attrs = self._graph.edge[l[0]][l[1]]
            for k in ['src', 'dest', 'features']:
                if k in attrs:
                    result[k] = attrs[k]
                else:
                    raise ValueError('Cannot use graph links without source, target, and features, run insertEnergies() first')
            return result

        traxelIdPerTimestepToUniqueIdMap, _ = self.getMappingsBetweenUUIDsAndTraxels()
        model = {
                'segmentationHypotheses':[translateNodeToDict(n) for n in self._graph.nodes_iter()],
                'linkingHypotheses':[translateLinkToDict(e) for e in self._graph.edges_iter()],
                'exclusions':[],
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
        # TODO: this recomputes the uuidToTraxelMap even though we have it already...
        trackingGraph = hytra.core.jsongraph.JsonTrackingGraph(model=model)

        return trackingGraph

    def insertSolution(self, resultDictionary):
        '''
        Add solution values to nodes and arcs from dictionary representation of solution.
        The resulting graph (=model) gets an additional property "value" that represents the number of objects inside a detection/arc
        Additionally a division indicator is saved in the node property "divisionValue".
        '''
        _, uuidToTraxelMap = self.getMappingsBetweenUUIDsAndTraxels()

        for detection in resultDictionary["detectionResults"]:
            self._graph.node[uuidToTraxelMap[detection["id"]][0]]['value'] = detection["value"]

        for link in resultDictionary["linkingResults"]:
            source, dest = uuidToTraxelMap[link["src"]][0], uuidToTraxelMap[link["dest"]][0]
            self._graph.edge[source][dest]['value'] = link["value"]

        for division in resultDictionary["divisionResults"]:
            self._graph.node[uuidToTraxelMap[division["id"]][0]]['divisionValue'] = division["value"]

    def countIncomingObjects(self,node):
        numberOfIncomingObject = 0
        numberOfIncomingEdges = 0
        for in_edge in self._graph.in_edges(node):
            numberOfIncomingObject += self._graph.node[node]['value']
            numberOfIncomingEdges += 1
        return numberOfIncomingObject,numberOfIncomingEdges

    def countOutgoingObjects(self,node):
        numberOfOutgoingObject = 0
        numberOfOutgoingEdges = 0
        for in_edge in self._graph.out_edges(node):
            numberOfOutgoingObject += self._graph.node[node]['value']
            numberOfOutgoingEdges += 1
        return numberOfOutgoingObject,numberOfOutgoingEdges

    def computeLineage(self):
        """
        computes lineage and track id for every node in the graph
        """

        update_queue = []
        max_lineage_id = 0
        max_track_id = 0
        # find start of lineages
        for n in self.nodeIterator():
            if self.countIncomingObjects(n)[0]==0:
                # found start of a track
                update_queue.append((n,max_lineage_id,max_track_id))
                max_lineage_id += 1
                max_track_id   += 1

        print update_queue

        while len(update_queue) > 0:
            current_node,lineage_id,track_id = update_queue.pop()
            self._graph.node[current_node]["lineageId"] = lineage_id
            self._graph.node[current_node]["trackId"] = track_id

            numberOfOutgoingObject,numberOfOutgoingEdges = self.countOutgoingObjects(current_node)
            print self._graph.node[current_node]
            if (numberOfOutgoingObject != numberOfOutgoingEdges):
                print "WARNING: running lineage computation on unresolved graphs depends on a race condition"

            if len(self._graph.out_edges(current_node)) == 1:
                a = self._graph.out_edges(current_node)[0]
                update_queue.append((self.target(a),
                                    lineage_id,
                                    track_id))
            elif len(self._graph.out_edges(current_node)) == 2:
                for a in self._graph.out_edges(current_node):
                    update_queue.append((self.target(a),
                                        lineage_id,
                                        max_track_id))
                    max_track_id += 1
