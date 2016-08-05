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
    
    def _readLabelImage(self, timeframe):
        '''
        Returns the labelimage for the given timeframe
        '''
        return self.labelVolume[timeframe, ..., 0]
    
    def _exportRefinedSegmentation(self, timesteps):
        """
        nothing to be done here, ilastik requests these images lazily 
        """
        pass

    def _computeObjectFeatures(self, timesteps):
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
