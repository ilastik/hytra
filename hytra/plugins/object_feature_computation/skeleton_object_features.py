from __future__ import print_function, absolute_import, nested_scopes, generators, division, with_statement, unicode_literals
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
        if 'Center' in featureDict:
            # old vigra versions simply call that feature "Center" which conflicts with other features 
            featureDict['Skeleton Center'] = featureDict['Center']
            del featureDict['Center']
        return featureDict
