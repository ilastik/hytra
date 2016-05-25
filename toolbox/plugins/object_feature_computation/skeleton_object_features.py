from toolbox.pluginsystem import object_feature_computation_plugin
import vigra
from vigra import numpy as np

class SkeletonObjectFeatures(object_feature_computation_plugin.ObjectFeatureComputationPlugin):
    """
    Computes the skeleton based vigra features
    """
    worksForDimensions = [2]
    omittedFeatures = ['Polygon']

    def computeFeatures(self, rawImage, labelImage, frameNumber, rawFilename):
        return vigra.analysis.extractSkeletonFeatures(labelImage.squeeze().astype(np.uint32))
