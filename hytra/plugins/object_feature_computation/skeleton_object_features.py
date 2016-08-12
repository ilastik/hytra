from hytra.pluginsystem import object_feature_computation_plugin
import vigra
from vigra import numpy as np

class SkeletonObjectFeatures(object_feature_computation_plugin.ObjectFeatureComputationPlugin):
    """
    Computes the skeleton based vigra features
    """
    worksForDimensions = [2]
    omittedFeatures = ['Polygon']

    def computeFeatures(self, rawImage, labelImage, frameNumber, rawFilename):
        featureDict = vigra.analysis.extractSkeletonFeatures(labelImage.squeeze().astype(np.uint32)) 
        featureDict['Skeleton Center'] = featureDict['Center']
        del featureDict['Center']
        return featureDict
