import os
import numpy as np
import logging
import hytra.core.mergerresolver
from hytra.core.probabilitygenerator import Traxel

def getLogger():
    ''' logger to be used in this module '''
    return logging.getLogger(__name__)

class IlastikMergerResolver(hytra.core.mergerresolver.MergerResolver):
    '''
    Specialization of merger resolving to work with the hypotheses graph given by ilastik,
    and to read/write images from/to the input/output slots of the respective operators. 
    '''
    def __init__(self, hypothesesGraph, labelVolume, pluginPaths=[os.path.abspath('../hytra/plugins')], verbose=False):
        super(IlastikMergerResolver, self).__init__(pluginPaths, verbose)
        trackingGraph = hypothesesGraph.toTrackingGraph(noFeatures=True)
        self.model = trackingGraph.model
        self.result = hypothesesGraph.getSolutionDictionary()
        self.hypothesesGraph = hypothesesGraph
        self.labelVolume = labelVolume
        self.relabeledVolume = np.zeros(labelVolume.shape, dtype=np.uint32)
    
    def _readLabelImage(self, timeframe):
        '''
        Returns the labelimage for the given timeframe
        '''
        return self.labelVolume[timeframe, ..., 0]
    
    def _exportRefinedSegmentation(self, labelImages):
        """
        Store the resulting label images, if needed.

        `labelImages` is a dictionary with str(timestep) as keys. 
        """
        for t, image in labelImages.iteritems():
            self.relabeledVolume[int(t), ..., 0] = image

    def _computeObjectFeatures(self, labelImages):
        '''
        Return the features per object as nested dictionaries:
        { (int(Timestep), int(Id)):{ "FeatureName" : np.array(value), "NextFeature": ...} }
        '''
        objectFeatures = {}

        # populate the dictionaries only with the Region Centers of the fit for the distance based
        # transitions in ilastik
        # TODO: in the future, this should recompute the object features from the relabeled image!
        for n in self.unresolvedGraph.nodes_iter():
            fits = self.unresolvedGraph.node[n]['fits']
            timestepIdTuples = [n]
            if 'newIds' in self.unresolvedGraph.node[n]:
                timestepIdTuples = [(n[0], i) for i in self.unresolvedGraph.node[n]['newIds']]
                assert(len(self.unresolvedGraph.node[n]['newIds']) == len(fits))

            for tidt, fit in zip(timestepIdTuples, fits):
                objectFeatures[tidt] = {'RegionCenter' : self._fitToRegionCenter(fit)}

        return objectFeatures
    
    def _fitToRegionCenter(self, fit):
        """
        Extract the region center from a GMM fit
        """
        return fit[2]
    
    def _refineResult(self,
                      nodeFlowMap,
                      arcFlowMap,
                      traxelIdPerTimestepToUniqueIdMap,
                      mergerNodeFilter,
                      mergerLinkFilter):
        """
        Overwrite parent method and simply call it, but then call _updateHypothesesGraph to
        also refine our Hypotheses Graph
        """
        refinedResult = super(IlastikMergerResolver, self)._refineResult(
            nodeFlowMap, arcFlowMap, traxelIdPerTimestepToUniqueIdMap, mergerNodeFilter, mergerLinkFilter)
        
        self._updateHypothesesGraph(arcFlowMap)

        return refinedResult

    def _updateHypothesesGraph(self, arcFlowMap):
        """
        After running merger resolving, insert new nodes, remove de-merged nodes
        and also update the links in the hypotheses graph.

        This also stores the new solution (`value` property) in the new nodes and links
        """
        # update nodes
        for n in self.unresolvedGraph.nodes_iter():
            # skip non-mergers
            if not 'newIds' in self.unresolvedGraph.node[n] or len(self.unresolvedGraph.node[n]['newIds']) < 2:
                continue
            
            # for this merger, insert all new nodes into the HG
            assert(len(self.unresolvedGraph.node[n]['newIds']) == self.unresolvedGraph.node[n]['count'])
            for newId, fit in zip(self.unresolvedGraph.node[n]['newIds'], self.unresolvedGraph.node[n]['fits']):
                traxel = Traxel()
                traxel.Id = newId
                traxel.Timestep = n[0]
                traxel.Features = self._fitToRegionCenter(fit)
                self.hypothesesGraph.addNodeFromTraxel(traxel, value=1)
            
            # remove merger from HG, which also removes all edges that would otherwise be dangling
            self.hypothesesGraph._graph.remove_node(n)

        # add new links
        for edge in self.resolvedGraph.edges_iter():
            srcId = self.resolvedGraph.node[edge[0]]['id']
            destId = self.resolvedGraph.node[edge[1]]['id']
            value = arcFlowMap[(srcId, destId)]
            self.hypothesesGraph._graph.add_edge(edge[0], edge[1], value=value)
    
    def relabelMergers(self, labelImage, time):
        """
        Calls the merger resolving plugin to relabel the mergers by fitting a gaussian mixture model.
        """
        
        nextObjectId = labelImage.max() + 1
        
        t = str(time)
        
        for idx in self.detectionsPerTimestep[t]:
            node = (time, idx)

            count = 1
            if idx in self.mergersPerTimestep[t]:
                count = self.mergersPerTimestep[t][idx]
            getLogger().debug("Looking at node {} in timestep {} with count {}".format(idx, t, count))
            
            # collect initializations from incoming
            initializations = []
            for predecessor, _ in self.unresolvedGraph.in_edges(node):
                initializations.extend(self.unresolvedGraph.node[predecessor]['fits'])
            # TODO: what shall we do if e.g. a 2-merger and a single object merge to 2 + 1,
            # so there are 3 initializations for the 2-merger, and two initializations for the 1 merger?
            # What does pgmlink do in that case?

            # use merger resolving plugin to fit `count` objects, also updates labelimage!
            fittedObjects = self.mergerResolverPlugin.resolveMerger(labelImage, idx, nextObjectId, count, initializations)
            assert(len(fittedObjects) == count)

            # split up node if count > 1, duplicate incoming and outgoing arcs
            if count > 1:
                nextObjectId += count
          
        return labelImage
