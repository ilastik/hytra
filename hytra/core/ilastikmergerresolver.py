import os
import numpy as np
import logging
import hytra.core.mergerresolver
from hytra.core.probabilitygenerator import Traxel
import hytra.core.probabilitygenerator


logger = logging.getLogger(__name__)


class IlastikMergerResolver(hytra.core.mergerresolver.MergerResolver):
    """
    Specialization of merger resolving to work with the hypotheses graph given by ilastik,
    and to read/write images from/to the input/output slots of the respective operators.
    """

    def __init__(
        self,
        hypothesesGraph,
        pluginPaths=[os.path.abspath("../hytra/plugins")],
        withFullGraph=False,
        numSplits=None,
        verbose=False,
        random_state=None,
    ):
        super(IlastikMergerResolver, self).__init__(pluginPaths, numSplits, verbose)
        trackingGraph = hypothesesGraph.toTrackingGraph(noFeatures=True)
        self.model = trackingGraph.model
        self.result = hypothesesGraph.getSolutionDictionary()
        self.hypothesesGraph = hypothesesGraph
        self._random_state = random_state

        # Find mergers in the given model and result
        traxelIdPerTimestepToUniqueIdMap, uuidToTraxelMap = hytra.core.jsongraph.getMappingsBetweenUUIDsAndTraxels(
            self.model
        )
        # there might be empty frames. We want them as output too.
        timesteps = [
            str(t)
            for t in range(
                int(min(traxelIdPerTimestepToUniqueIdMap.keys())),
                max([int(idx) for idx in traxelIdPerTimestepToUniqueIdMap.keys()]) + 1,
            )
        ]

        mergers, detections, links, divisions = hytra.core.jsongraph.getMergersDetectionsLinksDivisions(
            self.result, uuidToTraxelMap
        )

        self.mergerNum = len(mergers)

        # Check that graph contains mergers
        if self.mergerNum > 0:
            self.mergersPerTimestep = hytra.core.jsongraph.getMergersPerTimestep(mergers, timesteps)
            self.detectionsPerTimestep = hytra.core.jsongraph.getDetectionsPerTimestep(detections, timesteps)

            linksPerTimestep = hytra.core.jsongraph.getLinksPerTimestep(links, timesteps)
            divisionsPerTimestep = hytra.core.jsongraph.getDivisionsPerTimestep(divisions, linksPerTimestep, timesteps)
            mergerLinks = hytra.core.jsongraph.getMergerLinks(linksPerTimestep, self.mergersPerTimestep, timesteps)

            # Build graph of the unresolved (merger) nodes and their direct neighbors
            self._createUnresolvedGraph(
                divisionsPerTimestep,
                self.mergersPerTimestep,
                mergerLinks,
                withFullGraph,
            )
            self._prepareResolvedGraph()

    def run(self, transition_classifier_filename=None, transition_classifier_path=None):
        """
        Run merger resolving from within Ilastik
        We can't use run() from parent because it has to be done on a per frame basis.

        1. Compute object features
        2. Run min-cost max-flow tracking to find the fate of all the de-merged objects
        3. Export refined segmentation, update member variables `model` and `result`
        4. Compute merger dictionary

        **Returns** a nested dictionary, indexed first by time, then object Id, containing a list of new segmentIDs per merger
        """
        traxelIdPerTimestepToUniqueIdMap, uuidToTraxelMap = hytra.core.jsongraph.getMappingsBetweenUUIDsAndTraxels(
            self.model
        )
        # there might be empty frames. We want them as output too.
        timesteps = [
            str(t)
            for t in range(
                int(min(traxelIdPerTimestepToUniqueIdMap.keys())),
                max([int(idx) for idx in traxelIdPerTimestepToUniqueIdMap.keys()]) + 1,
            )
        ]

        # compute new object features
        objectFeatures = self._computeObjectFeatures(timesteps)

        # load transition classifier if any
        if transition_classifier_filename is not None:
            logger.info("\tLoading transition classifier")
            transitionClassifier = probabilitygenerator.RandomForestClassifier(
                transition_classifier_path, transition_classifier_filename
            )
        else:
            logger.info("\tUsing distance based transition energies")
            transitionClassifier = None

        # run min-cost max-flow to find merger assignments
        logger.info("Running min-cost max-flow to find resolved merger assignments")

        nodeFlowMap, arcFlowMap = self._minCostMaxFlowMergerResolving(objectFeatures, transitionClassifier)

        # fuse results into a new solution
        # 1.) replace merger nodes in JSON graph by their replacements -> new JSON graph
        #     update UUID to traxel map.
        #     a) how do we deal with the smaller number of states?
        #        Does it matter as we're done with tracking anyway..?

        def mergerNodeFilter(jsonNode):
            uuid = int(jsonNode["id"])
            traxels = uuidToTraxelMap[uuid]
            return not any(t[1] in self.mergersPerTimestep[str(t[0])] for t in traxels)

        def mergerLinkFilter(jsonLink):
            srcUuid = int(jsonLink["src"])
            destUuid = int(jsonLink["dest"])
            srcTraxels = uuidToTraxelMap[srcUuid]
            destTraxels = uuidToTraxelMap[destUuid]
            # return True if there was no traxel in either source or target node that was a merger.
            return not (
                any(t[1] in self.mergersPerTimestep[str(t[0])] for t in srcTraxels)
                or any(t[1] in self.mergersPerTimestep[str(t[0])] for t in destTraxels)
            )

        self.model = self._refineModel(
            uuidToTraxelMap,
            traxelIdPerTimestepToUniqueIdMap,
            mergerNodeFilter,
            mergerLinkFilter,
        )

        # 2.) new result = union(old result, resolved mergers) - old mergers
        self.result = self._refineResult(
            nodeFlowMap,
            arcFlowMap,
            traxelIdPerTimestepToUniqueIdMap,
            mergerNodeFilter,
            mergerLinkFilter,
        )

        # return a dictionary telling about which mergers were resolved into what
        mergerDict = {}
        for node in self.unresolvedGraph.nodes():
            # skip non-mergers
            if not "newIds" in self.unresolvedGraph.nodes[node] or len(self.unresolvedGraph.nodes[node]["newIds"]) < 2:
                continue

            # Save merger node info in merger dict (fits and new IDs used from within Ilastik)
            time = node[0]
            idx = node[1]
            mergerDict.setdefault(time, {})[idx] = self.unresolvedGraph.nodes[node]

        return mergerDict

    def getCoordinatesForObjectId(self, coordinatesForObjectIds, labelImage, timestep, objectId):
        """
        Get coordinate for object IDs in labelImage.
        """

        node = (timestep, objectId)

        mergerIsPresent = False
        if self.hypothesesGraph.hasNode(node):
            # Check if node is merger
            if (
                "value" in self.hypothesesGraph._graph.nodes[node]
                and self.hypothesesGraph._graph.nodes[node]["value"] > 1
            ):
                mergerIsPresent = True

            # Check if node is connected to merger
            if not mergerIsPresent:
                for edge in self.hypothesesGraph._graph.out_edges(node):
                    neighbor = edge[1]
                    if (
                        "value" in self.hypothesesGraph._graph.nodes[neighbor]
                        and self.hypothesesGraph._graph.nodes[neighbor]["value"] > 1
                    ):
                        mergerIsPresent = True

        # Compute coordinate for object ID
        if mergerIsPresent:
            coordinatesForObjectIds[objectId] = np.transpose(np.vstack(np.where(labelImage == objectId)))

    def fitAndRefineNodesForTimestep(self, coordinatesForObjectIds, maxObjectId, timestep):
        """
        Update segmentation of mergers (nodes in unresolvedGraph) for each frame
        and create new nodes in `resolvedGraph`. Links to merger nodes are duplicated to all new nodes.

        Uses the mergerResolver plugin to update the segmentations in the labelImages.

        This function is used by Ilastik to fit and refine nodes per frame instead of
        loading the full volume in _fitAndRefineNodes()
        """

        # use image provider plugin to load labelimage
        nextObjectId = maxObjectId + 1

        t = str(timestep)

        for idx, coordinates in sorted(coordinatesForObjectIds.items(), key=lambda x: x[0]):
            node = (timestep, idx)
            if node not in self.resolvedGraph:
                continue

            # Get merger count and initializations (only for merger nodes)
            count = 1
            initializations = []

            if idx in self.mergersPerTimestep[t]:
                count = self.mergersPerTimestep[t][idx]

                for predecessor, _ in self.unresolvedGraph.in_edges(node):
                    initializations.extend(self.unresolvedGraph.nodes[predecessor]["fits"])
                # TODO: what shall we do if e.g. a 2-merger and a single object merge to 2 + 1,
                # so there are 3 initializations for the 2-merger, and two initializations for the 1 merger?
                # What does pgmlink do in that case?

            logger.debug("Looking at node {} in timestep {} with count {}".format(idx, t, count))

            # use merger resolving plugin to fit `count` objects
            fittedObjects = list(
                self.mergerResolverPlugin.resolveMergerForCoords(
                    coordinates, count, initializations, random_state=self._random_state
                )
            )

            assert len(fittedObjects) == count

            # split up node if count > 1, duplicate incoming and outgoing arcs
            if count > 1:
                for idx in range(nextObjectId, nextObjectId + count):
                    newNode = (timestep, idx)
                    self.resolvedGraph.add_node(newNode, division=False, count=1, origin=node)

                    for e in self.unresolvedGraph.out_edges(node):
                        self.resolvedGraph.add_edge(newNode, e[1])
                    for e in self.unresolvedGraph.in_edges(node):
                        if "newIds" in self.unresolvedGraph.nodes[e[0]]:
                            for newId in self.unresolvedGraph.nodes[e[0]]["newIds"]:
                                self.resolvedGraph.add_edge((e[0][0], newId), newNode)
                        else:
                            self.resolvedGraph.add_edge(e[0], newNode)

                self.resolvedGraph.remove_node(node)
                self.unresolvedGraph.nodes[node]["newIds"] = range(nextObjectId, nextObjectId + count)
                nextObjectId += count

            # each unresolved node stores its fitted shape(s) to be used
            # as initialization in the next frame, this way division duplicates
            # and de-merged nodes in the resolved graph do not need to store a fit as well
            self.unresolvedGraph.nodes[node]["fits"] = fittedObjects

    def _computeObjectFeatures(self, timesteps):
        """
        Return the features per object as nested dictionaries:
        { (int(Timestep), int(Id)):{ "FeatureName" : np.array(value), "NextFeature": ...} }
        """
        objectFeatures = {}

        # populate the dictionaries only with the Region Centers of the fit for the distance based
        # transitions in ilastik
        # TODO: in the future, this should recompute the object features from the relabeled image!
        for node in self.unresolvedGraph.nodes():
            # Add region centers for new nodes (based on GMM fits)
            if "newIds" in self.unresolvedGraph.nodes[node] and "fits" in self.unresolvedGraph.nodes[node]:
                assert len(self.unresolvedGraph.nodes[node]["newIds"]) == len(self.unresolvedGraph.nodes[node]["fits"])

                time = node[0]
                newNodes = [(time, idx) for idx in self.unresolvedGraph.nodes[node]["newIds"]]
                fits = self.unresolvedGraph.nodes[node]["fits"]

                for newNode, fit in zip(newNodes, fits):
                    objectFeatures[newNode] = {"RegionCenter": self._fitToRegionCenter(fit)}

            # Otherwise, get the region centers from the traxel com feature
            else:
                objectFeatures[node] = {
                    "RegionCenter": self.hypothesesGraph._graph.nodes[node]["traxel"].Features["com"]
                }

        return objectFeatures

    def _fitToRegionCenter(self, fit):
        """
        Extract the region center from a GMM fit
        """
        return fit[2]

    def _refineResult(
        self,
        nodeFlowMap,
        arcFlowMap,
        traxelIdPerTimestepToUniqueIdMap,
        mergerNodeFilter,
        mergerLinkFilter,
    ):
        """
        Overwrite parent method and simply call it, but then call _updateHypothesesGraph to
        also refine our Hypotheses Graph
        """
        refinedResult = super(IlastikMergerResolver, self)._refineResult(
            nodeFlowMap,
            arcFlowMap,
            traxelIdPerTimestepToUniqueIdMap,
            mergerNodeFilter,
            mergerLinkFilter,
        )

        self._updateHypothesesGraph(arcFlowMap)

        return refinedResult

    def _updateHypothesesGraph(self, arcFlowMap):
        """
        After running merger resolving, insert new nodes, remove de-merged nodes
        and also update the links in the hypotheses graph.

        This also stores the new solution (`value` property) in the new nodes and links
        """

        # update nodes
        for n in self.unresolvedGraph.nodes():
            # skip non-mergers
            if not "newIds" in self.unresolvedGraph.nodes[n] or len(self.unresolvedGraph.nodes[n]["newIds"]) < 2:
                continue

            # for this merger, insert all new nodes into the HG
            assert len(self.unresolvedGraph.nodes[n]["newIds"]) == self.unresolvedGraph.nodes[n]["count"]
            for newId, fit in zip(
                self.unresolvedGraph.nodes[n]["newIds"],
                self.unresolvedGraph.nodes[n]["fits"],
            ):
                traxel = Traxel()
                traxel.Id = newId
                traxel.Timestep = n[0]

                traxel.Features = {"com": self._fitToRegionCenter(fit)}
                self.hypothesesGraph.addNodeFromTraxel(traxel, value=1, mergerValue=n[1], divisionValue=False)

            # remove merger from HG, which also removes all edges that would otherwise be dangling
            self.hypothesesGraph._graph.remove_node(n)

        # add new links only for merger nodes
        for edge in self.resolvedGraph.edges():
            # Add new edges that are connected to new merger nodes
            if (
                "mergerValue" in self.hypothesesGraph._graph.nodes[edge[0]]
                or "mergerValue" in self.hypothesesGraph._graph.nodes[edge[1]]
            ):
                srcId = self.resolvedGraph.nodes[edge[0]]["id"]
                destId = self.resolvedGraph.nodes[edge[1]]["id"]

                edgeValue = arcFlowMap[(srcId, destId)]

                # edges connected to mergers are set to "not used" in order to prevent multiple active outgoing edges from single nodes. The correct edges will be added later.
                if edgeValue > 0 and not self.resolvedGraph.nodes[edge[0]]["division"]:
                    for outEdge in self.hypothesesGraph._graph.out_edges(edge[0]):
                        self.hypothesesGraph._graph.edges[outEdge[0], outEdge[1]]["value"] = 0

                    for inEdge in self.hypothesesGraph._graph.in_edges(edge[1]):
                        self.hypothesesGraph._graph.edges[inEdge[0], inEdge[1]]["value"] = 0

                # Add new edge connected to merger node
                self.hypothesesGraph._graph.add_edge(edge[0], edge[1], value=edgeValue)
