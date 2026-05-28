from typing import Type

from hytra.types.object_feature_computation import ObjectFeatureComputationBase
from hytra.types.transition_feature_vector_construction import TransitionFeatureVectorConstructionBase


def apply_object_feature_computation(ndims, rawImage, labelImage, frameNumber, rawFilename, feature_ops: list[Type[ObjectFeatureComputationBase]]):
    """
    computes the features of all plugins and returns a list of dictionaries, as well as a list of
    feature names that should be ignored
    """
    features = []
    featureNamesToIgnore = []

    for feature_cls in feature_ops:
        feature_ins = feature_cls()
        if ndims in feature_ins.worksForDimensions:
            f = feature_ins.computeFeatures(rawImage, labelImage, frameNumber, rawFilename)
            features.append(f)
            featureNamesToIgnore.extend(feature_ins.omittedFeatures)

    return features, featureNamesToIgnore


def apply_transition_feature_vector_construction(featureDictObjectA, featureDictObjectB, selectedFeatures, feature_ops: list[Type[TransitionFeatureVectorConstructionBase]]):
    """
    constructs a transition feature vector for training/prediction with a random forest from the
    features of the two objects participating in the transition.
    """
    featureVector = []

    for feature_cls in feature_ops:
        feature_ins = feature_cls()
        f = feature_ins.constructFeatureVector(featureDictObjectA, featureDictObjectB, selectedFeatures)
        featureVector.extend(f)

    return featureVector


def get_transition_feature_names(featureDictObjectA, featureDictObjectB, selectedFeatures, feature_ops: list[Type[TransitionFeatureVectorConstructionBase]]):
    """
    returns a verbal description of each feature in the transition feature vector
    """
    featureNames = []

    for feature_cls in feature_ops:
        feature_ins = feature_cls()
        f = feature_ins.getFeatureNames(featureDictObjectA, featureDictObjectB, selectedFeatures)
        featureNames.extend(f)

    return featureNames
