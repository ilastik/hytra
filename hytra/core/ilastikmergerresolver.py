import os
import logging
import hytra.core.mergerresolver

def getLogger():
    ''' logger to be used in this module '''
    return logging.getLogger(__name__)

class IlastikMergerResolver(hytra.core.mergerresolver.MergerResolver):
    '''
    Specialization of merger resolving to work with the hypotheses graph given by ilastik,
    and to read/write images from/to the input/output slots of the respective operators. 
    '''
    def __init__(self, hypothesesGraph, pluginPaths=[os.path.abspath('../hytra/plugins')], verbose=False):
        super(IlastikMergerResolver, self).__init__(pluginPaths, verbose)
        trackingGraph = hypothesesGraph.toTrackingGraph()
        self.model = trackingGraph.model
        self.result = trackingGraph.result
    
    def _readLabelImage(self, timeframe):
        '''
        Returns the labelimage for the given timeframe
        '''
        raise NotImplementedError()
    
    def _exportRefinedSegmentation(self, labelImages):
        """
        Store the resulting label images, if needed.

        `labelImages` is a dictionary with str(timestep) as keys. 
        """
        raise NotImplementedError()

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