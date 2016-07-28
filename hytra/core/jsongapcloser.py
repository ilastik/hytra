import copy
import logging
import h5py
import os
import hytra.core.gapcloser
from hytra.core.jsongraph import JsonTrackingGraph

def getLogger():
    ''' logger to be used in this module '''
    return logging.getLogger(__name__)

class JsonGapCloser(hytra.core.gapcloser.GapCloser):
    '''
    Specialization of gap closing that reads raw data and segmentation from HDF5 files,
    computes features on demand, and that uses JSON files as input/output for graph and tracking result.
    '''
    def __init__(self,
                 jsonTrackingGraph,
                 label_image_filename,
                 label_image_path,
                 raw_filename,
                 raw_path,
                 raw_axes,
                 pluginPaths=[os.path.abspath('../hytra/plugins')],
                 verbose=False):
        super(JsonGapCloser, self).__init__(pluginPaths, verbose)

        # copy model and result because we will modify it here
        assert(isinstance(jsonTrackingGraph, JsonTrackingGraph))
        assert(jsonTrackingGraph.model is not None and len(jsonTrackingGraph.model) > 0)
        assert(jsonTrackingGraph.result is not None and len(jsonTrackingGraph.result) > 0)
        self.model = copy.copy(jsonTrackingGraph.model)
        self.result = copy.copy(jsonTrackingGraph.result)

        assert(self.result['detectionResults'] is not None)
        assert(self.result['linkingResults'] is not None)
        self.withDivisions = self.result['divisionResults'] is not None

        self.label_image_filename = label_image_filename
        self.label_image_path = label_image_path
        self.raw_filename = raw_filename
        self.raw_path = raw_path
        self.raw_axes = raw_axes
        self.pluginManager.setImageProvider('LocalImageLoader')
        self.imageProvider = self.pluginManager.getImageProvider()
    

    def _computeObjectFeatures(self, labelImages):
        """
        Computes object features for all nodes in the resolved graph because they
        are needed for the transition classifier or to compute new distances.

        **returns:** a dictionary of feature-dicts per node
        """
        rawImages = {}
        for t in labelImages.keys():
            rawImages[t] = self.imageProvider.getImageDataAtTimeFrame(self.raw_filename, self.raw_path, self.raw_axes, int(t))

        getLogger().info("Computing object features")
        objectFeatures = {}
        imageShape = self.imageProvider.getImageShape(self.label_image_filename, self.label_image_path)
        getLogger().info("Found image of shape {}".format(imageShape))
        # ndims = len(np.array(imageShape).squeeze()) - 1 # get rid of axes with length 1, and minus time axis
        # there is no time axis...
        ndims = len([i for i in imageShape if i != 1])
        getLogger().info("Data has dimensionality {}".format(ndims))
        for node in self.Graph.nodes_iter():
            intT, idx = node
            if isinstance(idx, str) and idx.startswith('div-'):
                continue

            # mask out this object only and compute features
            mask = labelImages[str(intT)].copy()
            mask[mask != idx] = 0
            mask[mask == idx] = 1

            # compute features, transform to one dict for frame
            frameFeatureDicts, ignoreNames = self.pluginManager.applyObjectFeatureComputationPlugins(
                ndims, rawImages[str(intT)], mask, intT, self.raw_filename)
            frameFeatureItems = []
            for f in frameFeatureDicts:
                frameFeatureItems = frameFeatureItems + f.items()
            frameFeatures = dict(frameFeatureItems)

            # extract all features for this one object
            objectFeatureDict = {}
            for k, v in frameFeatures.iteritems():
                if k in ignoreNames:
                    continue
                elif 'Polygon' in k:
                    objectFeatureDict[k] = v[1]
                else:
                    objectFeatureDict[k] = v[1, ...]
            objectFeatures[node] = objectFeatureDict

        return objectFeatures
    
    def _readLabelImage(self, timeframe):
        '''
        Returns the labelimage for the given timeframe
        '''
        return self.imageProvider.getLabelImageForFrame(self.label_image_filename, self.label_image_path, timeframe)