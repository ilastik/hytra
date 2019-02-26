import copy
import logging
import h5py
import numpy as np
import os
import hytra.core.mergerresolver
from hytra.core.jsongraph import JsonTrackingGraph


logger = logging.getLogger(__name__)


class JsonMergerResolver(hytra.core.mergerresolver.MergerResolver):
    '''
    Specialization of merger resolving that reads raw data and segmentation from HDF5 files,
    computes features on demand, and that uses JSON files as input/output for graph and tracking result.
    '''
    def __init__(self,
                 jsonTrackingGraph,
                 label_image_filename,
                 label_image_path,
                 out_label_image,
                 raw_filename,
                 raw_path,
                 raw_axes,
                 pluginPaths=[os.path.abspath('../hytra/plugins')],
                 verbose=False):
        super(JsonMergerResolver, self).__init__(pluginPaths, verbose=verbose)

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
        self.out_label_image = out_label_image
        self.raw_filename = raw_filename
        self.raw_path = raw_path
        self.raw_axes = raw_axes
        self.pluginManager.setImageProvider('LocalImageLoader')
        self.imageProvider = self.pluginManager.getImageProvider()


    def _computeObjectFeatures(self, timesteps):
        """
        Computes object features for all nodes in the resolved graph because they
        are needed for the transition classifier or to compute new distances.

        **returns:** a dictionary of feature-dicts per node
        """
        rawImages = {}
        labelImages = {}
        for t in timesteps:
            if self.raw_filename is not None:
                rawImages[str(t)] = self.imageProvider.getImageDataAtTimeFrame(self.raw_filename, self.raw_path, self.raw_axes, int(t))
            else:
                rawImages[str(t)] = self.imageProvider.getLabelImageForFrame(self.label_image_filename, self.label_image_path, int(t)).astype(np.float32)
            labelImages[str(t)] = self.imageProvider.getLabelImageForFrame(self.label_image_filename, self.label_image_path, int(t))
            self.relabelMergers(labelImages[str(t)], int(t))

        logger.info("Computing object features")
        objectFeatures = {}
        imageShape = self.imageProvider.getImageShape(self.label_image_filename, self.label_image_path)
        logger.info("Found image of shape {}".format(imageShape))
        # ndims = len(np.array(imageShape).squeeze()) - 1 # get rid of axes with length 1, and minus time axis
        # there is no time axis...
        ndims = len([i for i in imageShape if i != 1])
        logger.info("Data has dimensionality {}".format(ndims))
        for node in self.resolvedGraph.nodes_iter():
            intT, idx = node
            if str(idx).startswith('div-'):
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
                frameFeatureItems = frameFeatureItems + list(f.items())
            frameFeatures = dict(frameFeatureItems)

            # extract all features for this one object
            objectFeatureDict = {}
            for k, v in frameFeatures.items():
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

    def _exportRefinedSegmentation(self, timesteps):
        h5py.File(self.out_label_image, 'w').close()
        for t in timesteps:
            labelImage = self._readLabelImage(int(t))
            self.relabelMergers(labelImage, int(t))
            self.imageProvider.exportLabelImage(labelImage, int(t), self.out_label_image, self.label_image_path)
