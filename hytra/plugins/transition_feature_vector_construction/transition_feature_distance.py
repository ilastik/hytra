from hytra.pluginsystem import transition_feature_vector_construction_plugin
import numpy as np


class TransitionFeaturesDistance(transition_feature_vector_construction_plugin.TransitionFeatureVectorConstructionPlugin):
    """
    Computes distance based features for transitions
    """

    def constructFeatureVector(self, featureDictObjectA, featureDictObjectB, selectedFeatures):
        key = 'RegionCenter'
        if key in selectedFeatures:
            return [np.linalg.norm(featureDictObjectA[key] - featureDictObjectB[key]),
                    np.linalg.norm(featureDictObjectA[key] * featureDictObjectB[key])]
        return []

    def getFeatureNames(self, featureDictObjectA, featureDictObjectB, selectedFeatures):
        key = 'RegionCenter'
        if key in selectedFeatures:
            return ['norm(A[RegionCenter]-B[RegionCenter]), norm(A[RegionCenter]*B[RegionCenter])']
        return []

