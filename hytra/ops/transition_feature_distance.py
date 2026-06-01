import numpy as np

from ..types.transition_feature_vector_construction import TransitionFeatureVectorConstructionBase


class TransitionFeaturesDistance(TransitionFeatureVectorConstructionBase):
    """
    Computes distance based features for transitions
    """

    def constructFeatureVector(self, featureDictObjectA, featureDictObjectB, selectedFeatures):
        key = "RegionCenter"
        if key in selectedFeatures:
            return [
                np.linalg.norm(featureDictObjectA[key] - featureDictObjectB[key]),
                np.linalg.norm(featureDictObjectA[key] * featureDictObjectB[key]),
            ]
        return []

    def getFeatureNames(self, featureDictObjectA, featureDictObjectB, selectedFeatures):
        key = "RegionCenter"
        if key in selectedFeatures:
            return ["norm(A[RegionCenter]-B[RegionCenter]), norm(A[RegionCenter]*B[RegionCenter])"]
        return []
