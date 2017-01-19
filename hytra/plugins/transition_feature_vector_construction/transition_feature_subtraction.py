from __future__ import unicode_literals
from builtins import range
from hytra.pluginsystem import transition_feature_vector_construction_plugin
import numpy as np
from compiler.ast import flatten


class TransitionFeaturesSubtraction(transition_feature_vector_construction_plugin.TransitionFeatureVectorConstructionPlugin):
    """
    Computes the subtraction of features in the feature vector
    """

    def constructFeatureVector(self, featureDictObjectA, featureDictObjectB, selectedFeatures):
        assert ("Global<Maximum >" not in selectedFeatures)
        assert ("Global<Minimum >" not in selectedFeatures)
        assert ("Histrogram" not in selectedFeatures)
        assert ("Polygon" not in selectedFeatures)

        features = []

        for key in selectedFeatures:
            if key == 'RegionCenter':
                continue
            else:
                if not isinstance(featureDictObjectA[key], np.ndarray) or featureDictObjectA[key].size == 1:
                    features.append(float(featureDictObjectA[key]) - float(featureDictObjectB[key]))
                else:
                    features.extend(flatten((featureDictObjectA[key].astype('float32') \
                                             - featureDictObjectB[key].astype('float32')).tolist()))

        # there should be no nans or infs
        assert (np.all(np.isfinite(np.array(features))))

        return features

    def getFeatureNames(self, featureDictObjectA, featureDictObjectB, selectedFeatures):
        assert ("Global<Maximum >" not in selectedFeatures)
        assert ("Global<Minimum >" not in selectedFeatures)
        assert ("Histrogram" not in selectedFeatures)
        assert ("Polygon" not in selectedFeatures)

        featuresNames = []

        for key in selectedFeatures:
            if key == 'RegionCenter':
                continue
            else:
                if not isinstance(featureDictObjectA[key], np.ndarray) or featureDictObjectA[key].size == 1:
                    featuresNames.append('A[{key}]-B[{key}]'.format(key=key))
                else:
                    featuresNames.extend(
                        ['A[{key}][{i}]-B[{key}][{i}]'.format(key=key, i=i) 
                         for i in range(len((featureDictObjectA[key] - featureDictObjectB[key]).tolist()))])

        return featuresNames

