from toolbox.pluginsystem import object_feature_computation_plugin
import vigra
from vigra import numpy as np

class StandardObjectFeatures(object_feature_computation_plugin.ObjectFeatureComputationPlugin):
    """
    Computes the standard vigra region features
    """
    worksForDimensions = [2, 3]
    omittedFeatures = ["Global<Maximum >", "Global<Minimum >", 'Histogram', 'Weighted<RegionCenter>']

    def computeFeatures(self, rawImage, labelImage, frameNumber, rawFilename):
        return vigra.analysis.extractRegionFeatures(rawImage.squeeze().astype('float32'),
                                                    labelImage.squeeze().astype('uint32'),
                                                    ignoreLabel=0)

