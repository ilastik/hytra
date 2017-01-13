from __future__ import print_function, absolute_import, nested_scopes, generators, division, with_statement, unicode_literals
from hytra.pluginsystem import object_feature_computation_plugin
import vigra
from vigra import numpy as np

class ConvexHullObjectFeatures(object_feature_computation_plugin.ObjectFeatureComputationPlugin):
    """
    Computes the convex hull based vigra features
    """
    worksForDimensions = [2]
    omittedFeatures = ['Polygon']

    def computeFeatures(self, rawImage, labelImage, frameNumber, rawFilename):
        featureDict =  vigra.analysis.extractConvexHullFeatures(labelImage.squeeze().astype(np.uint32),
                                                                ignoreLabel=0)
        if 'Center' in featureDict:
            # old vigra versions simply call that feature "Center" which conflicts with other features 
            featureDict['Hull Center'] = featureDict['Center']
            del featureDict['Center']
        return featureDict

