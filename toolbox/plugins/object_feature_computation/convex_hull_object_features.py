from pluginsystem import object_feature_computation_plugin
import vigra
from vigra import numpy as np

class ConvexHullObjectFeatures(object_feature_computation_plugin.ObjectFeatureComputationPlugin):
    """
    Computes the convex hull based vigra features
    """
    worksForDimensions = [2]
    omittedFeatures = ['Polygon', 'Defect Center', 'Center', 'Input Center']

    def computeFeatures(self, rawImage, labelImage, frameNumber, rawFilename):
        return vigra.analysis.extractConvexHullFeatures(labelImage.squeeze().astype(np.uint32),
                                                        ignoreLabel=0)

