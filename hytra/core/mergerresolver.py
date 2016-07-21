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

class MergerResolver(object):
    """
    Base class for all merger resolving implementations. Use one of the derived classes
    that handle reading/writing data to the respective sources.
    """

    def __init__(self, pluginPaths=[os.path.abspath('../hytra/plugins')], verbose=False):
        self.unresolvedGraph = None
        self.resolvedGraph = None
        self.pluginManager = TrackingPluginManager(verbose=verbose, pluginPaths=pluginPaths)

        # should be filled by constructors of derived classes!
        self.model = None
        self.result = None

    def _createUnresolvedGraph(self, divisionsPerTimestep, mergersPerTimestep, mergerLinks):
        """
        Set up a networkx graph consisting of mergers that need to be resolved (not resolved yet!)
        and their direct neighbors.

        ** returns ** the `unresolvedGraph`
        """
        self.unresolvedGraph = nx.DiGraph()
        def source(timestep, link):
            return int(timestep) - 1, link[0]

        def target(timestep, link):
            return int(timestep), link[1]

        def addNode(node):
            ''' add a node to the unresolved graph and fill in the properties `division` and `count` '''
            intT, idx = node

            lastframe = max(divisionsPerTimestep.keys(), key=int)
            if divisionsPerTimestep is not None and int(intT)<int(lastframe):
                division = idx in divisionsPerTimestep[str(intT+1)] # +1 screams for lastframe condition.
            else:
                division = False
            count = 1
            if idx in mergersPerTimestep[str(intT)]:
                assert(not division)
                count = mergersPerTimestep[str(intT)][idx]
            self.unresolvedGraph.add_node(node, division=division, count=count)

        # add nodes
        for t, link in mergerLinks:
            for n in [source(t, link), target(t, link)]:
                if not self.unresolvedGraph.has_node(n):
                    addNode(n)
            self.unresolvedGraph.add_edge(source(t, link), target(t,link))

        return self.unresolvedGraph

    def _prepareResolvedGraph(self): 
        """
        Split up the division nodes in the `unresolvedGraph` if they have two outgoing links
        and add the duplicates to the resolved graph, which is apart from that a copy of
        the `unresolvedGraph`.

        ** returns ** the `resolvedGraph`
        """
        self.resolvedGraph = self.unresolvedGraph.copy()
        numDivisionNodes = 0
        for n in self.unresolvedGraph:
            if self.unresolvedGraph.node[n]['division'] and len(self.unresolvedGraph.out_edges(n)) == 2:
                # create a duplicate node, make one link start from there
                duplicate = (n[0], 'div-{}'.format(numDivisionNodes))
                numDivisionNodes += 1
                self.resolvedGraph.add_node(duplicate, division=False, count=1)

                dest = self.unresolvedGraph.out_edges(n)[0]
                self.resolvedGraph.add_edge(duplicate, dest)
                self.resolvedGraph.remove_edge(n, dest)

                # store node references
                self.resolvedGraph.node[duplicate]['origin'] = n
                self.resolvedGraph.node[n]['duplicate'] = duplicate

        return self.resolvedGraph

    def _readLabelImage(self, timeframe):
        '''
        Should return the labelimage for the given timeframe
        '''
        raise NotImplementedError()

    def _refineSegmentation(self,
                            detectionsPerTimestep,
                            mergersPerTimestep,
                            timesteps):
        '''
        Update segmentation of mergers (nodes in unresolvedGraph) from first timeframe to last
        and create new nodes in `resolvedGraph`. Links to merger nodes are duplicated to all new nodes.

        Uses the mergerResolver plugin to update the segmentations in the labelImages.
        '''

        mergerResolver = self.pluginManager.getMergerResolver()

        intTimesteps = [int(t) for t in timesteps]
        intTimesteps.sort()

        labelImages = {}

        for intT in intTimesteps:
            t = str(intT)
            # use image provider plugin to load labelimage
            labelImage = self._readLabelImage(int(t))
            labelImages[t] = labelImage
            nextObjectId = labelImage.max() + 1

            for idx in detectionsPerTimestep[t]:
                node = (intT, idx)
                if node not in self.resolvedGraph:
                    continue

                count = 1
                if idx in mergersPerTimestep[t]:
                    count = mergersPerTimestep[t][idx]
                getLogger().debug("Looking at node {} in timestep {} with count {}".format(idx, t, count))
                
                # collect initializations from incoming
                initializations = []
                for predecessor, _ in self.unresolvedGraph.in_edges(node):
                    initializations.extend(self.unresolvedGraph.node[predecessor]['fits'])
                # TODO: what shall we do if e.g. a 2-merger and a single object merge to 2 + 1, 
                # so there are 3 initializations for the 2-merger, and two initializations for the 1 merger?
                # What does pgmlink do in that case?

                # use merger resolving plugin to fit `count` objects, also updates labelimage!
                fittedObjects = mergerResolver.resolveMerger(labelImage, idx, nextObjectId, count, initializations)
                assert(len(fittedObjects) == count)

                # split up node if count > 1, duplicate incoming and outgoing arcs
                if count > 1:
                    for idx in range(nextObjectId, nextObjectId + count):
                        newNode = (intT, idx)
                        self.resolvedGraph.add_node(newNode, division=False, count=1, origin=node)
                        
                        for e in self.unresolvedGraph.out_edges(node):
                            self.resolvedGraph.add_edge(newNode, e[1])
                        for e in self.unresolvedGraph.in_edges(node):
                            if 'newIds' in self.unresolvedGraph.node[e[0]]:
                                for newId in self.unresolvedGraph.node[e[0]]['newIds']:
                                    self.resolvedGraph.add_edge((e[0][0], newId), newNode)
                            else:
                                self.resolvedGraph.add_edge(e[0], newNode)

                    self.resolvedGraph.remove_node(node)
                    self.unresolvedGraph.node[node]['newIds'] = range(nextObjectId, nextObjectId + count)
                    nextObjectId += count

                # each unresolved node stores its fitted shape(s) to be used 
                # as initialization in the next frame, this way division duplicates 
                # and de-merged nodes in the resolved graph do not need to store a fit as well
                self.unresolvedGraph.node[node]['fits'] = fittedObjects

        # import matplotlib.pyplot as plt
        # nx.draw_networkx(resolvedGraph)
        # plt.savefig("/Users/chaubold/test.pdf")

        return labelImages

    def _minCostMaxFlowMergerResolving(self, objectFeatures, transitionClassifier=None, transitionParameter=5.0):
        """
        Find the optimal assignments within the `resolvedGraph` by running min-cost max-flow from the
        `dpct` module.

        Converts the `resolvedGraph` to our JSON model structure, predicts the transition probabilities 
        either using the given transitionClassifier, or using distance-based probabilities.

        **returns** a `nodeFlowMap` and `arcFlowMap` holding information on the usage of the respective nodes and links

        **Note:** cannot use `networkx` flow methods because they don't work with floating point weights.
        """

        trackingGraph = JsonTrackingGraph()
        for node in self.resolvedGraph.nodes_iter():
            additionalFeatures = {}
            if len(self.resolvedGraph.in_edges(node)) == 0:
                additionalFeatures['appearanceFeatures'] = [[0], [0]]
            if len(self.resolvedGraph.out_edges(node)) == 0:
                additionalFeatures['disappearanceFeatures'] = [[0], [0]]
            uuid = trackingGraph.addDetectionHypotheses([[0], [1]], **additionalFeatures)
            self.resolvedGraph.node[node]['id'] = uuid

        for edge in self.resolvedGraph.edges_iter():
            src = self.resolvedGraph.node[edge[0]]['id']
            dest = self.resolvedGraph.node[edge[1]]['id']

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

        return nodeFlowMap, arcFlowMap

    def _refineModel(self,
                     uuidToTraxelMap, 
                     traxelIdPerTimestepToUniqueIdMap, 
                     mergerNodeFilter, 
                     mergerLinkFilter):
        """
        Take the `self.model` (JSON format) with mergers, remove the merger nodes, but add new
        de-merged nodes and links. Also updates `traxelIdPerTimestepToUniqueIdMap` locally and in the resulting file,
        such that the traxel IDs match the new connected component IDs in the refined images.

        `mergerNodeFilter` and `mergerLinkFilter` are methods that can filter merger detections
        and links from the respective lists in the `model` dict.
        
        **Returns** the updated `model` dictionary, which is the same as the input `model` (works in-place)
        """

        # remove merger detections
        self.model['segmentationHypotheses'] = filter(mergerNodeFilter, self.model['segmentationHypotheses'])

        # remove merger links
        self.model['linkingHypotheses'] = filter(mergerLinkFilter, self.model['linkingHypotheses'])

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
                    self.model['segmentationHypotheses'].append(newDetection)
                    traxelIdPerTimestepToUniqueIdMap[str(node[0])][str(newId)] = nextUuid
                    nextUuid += 1

        # insert new links
        for edge in self.resolvedGraph.edges_iter():
            newLink = {}
            newLink['src'] = traxelIdPerTimestepToUniqueIdMap[str(edge[0][0])][str(edge[0][1])]
            newLink['dest'] = traxelIdPerTimestepToUniqueIdMap[str(edge[1][0])][str(edge[1][1])]
            self.model['linkingHypotheses'].append(newLink)

        # save
        return self.model

    def _refineResult(self,
                      nodeFlowMap,
                      arcFlowMap,
                      traxelIdPerTimestepToUniqueIdMap,
                      mergerNodeFilter,
                      mergerLinkFilter):
        """
        Update the `self.result` dict by removing the mergers and adding the refined nodes and links.

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
        self.result['detectionResults'] = [r for r in self.result['detectionResults'] if mergerNodeFilter(r)]
        self.result['linkingResults'] = [r for r in self.result['linkingResults'] if mergerLinkFilter(r)]

        # add new nodes
        for node in self.unresolvedGraph.nodes_iter():
            if self.unresolvedGraph.node[node]['count'] > 1:
                newIds = self.unresolvedGraph.node[node]['newIds']
                for newId in newIds:
                    uuid = traxelIdPerTimestepToUniqueIdMap[str(node[0])][str(newId)]
                    resolvedNode = (node[0], newId)
                    resolvedResultId = self.resolvedGraph.node[resolvedNode]['id']
                    newDetection = {'id': uuid, 'value': nodeFlowMap[resolvedResultId]}
                    self.result['detectionResults'].append(newDetection)

        # add new links
        for edge in self.resolvedGraph.edges_iter():
            newLink = {}
            newLink['src'] = traxelIdPerTimestepToUniqueIdMap[str(edge[0][0])][str(edge[0][1])]
            newLink['dest'] = traxelIdPerTimestepToUniqueIdMap[str(edge[1][0])][str(edge[1][1])]
            srcId = self.resolvedGraph.node[edge[0]]['id']
            destId = self.resolvedGraph.node[edge[1]]['id']
            newLink['value'] = arcFlowMap[(srcId, destId)]
            self.result['linkingResults'].append(newLink)

        return self.result

    def _exportRefinedSegmentation(self, labelImages):
        """
        Store the resulting label images, if needed.

        `labelImages` is a dictionary with str(timestep) as keys. 
        """
        pass

    def _computeObjectFeatures(self, labelImages):
        '''
        Return the features per object as nested dictionaries:
        { (int(Timestep), int(Id)):{ "FeatureName" : np.array(value), "NextFeature": ...} }
        '''
        pass

    # ------------------------------------------------------------
    def run(self, transition_classifier_filename=None, transition_classifier_path=None):
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


        # ------------------------------------------------------------

        # it may be, that there are no mergers, so do basically nothing, just copy all the ingoing data
        if len(mergers) == 0:
            getLogger().info("The maximum number of objects is 1, so nothing to be done. Writing the output...")
            # segmentation

            labelImages = {}
            for t in timesteps:
                # use image provider plugin to load labelimage
                labelImage = self._readLabelImage(int(t))
                labelImages[t] = labelImage
            self._exportRefinedSegmentation(labelImages)

        else:
            mergersPerTimestep = hytra.core.jsongraph.getMergersPerTimestep(mergers, timesteps)
            linksPerTimestep = hytra.core.jsongraph.getLinksPerTimestep(links, timesteps)
            detectionsPerTimestep = hytra.core.jsongraph.getDetectionsPerTimestep(detections, timesteps)
            divisionsPerTimestep = hytra.core.jsongraph.getDivisionsPerTimestep(divisions, linksPerTimestep, timesteps)
            mergerLinks = hytra.core.jsongraph.getMergerLinks(linksPerTimestep, mergersPerTimestep, timesteps)
            
            # set up unresolved graph and then refine the nodes to get the resolved graph
            self._createUnresolvedGraph(divisionsPerTimestep, mergersPerTimestep, mergerLinks)
            self._prepareResolvedGraph()
            labelImages = self._refineSegmentation(detectionsPerTimestep,
                                                   mergersPerTimestep,
                                                   timesteps)

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
            # run min-cost max-flow to find merger assignments
            getLogger().info("Running min-cost max-flow to find resolved merger assignments")

            nodeFlowMap, arcFlowMap = self._minCostMaxFlowMergerResolving(objectFeatures, transitionClassifier)

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

            self.model = self._refineModel(uuidToTraxelMap,
                                           traxelIdPerTimestepToUniqueIdMap,
                                           mergerNodeFilter,
                                           mergerLinkFilter)

            # 2.) new result = union(old result, resolved mergers) - old mergers

            self.result = self._refineResult(nodeFlowMap,
                                             arcFlowMap,
                                             traxelIdPerTimestepToUniqueIdMap,
                                             mergerNodeFilter,
                                             mergerLinkFilter)

            # 3.) export refined segmentation
            self._exportRefinedSegmentation(labelImages)
