from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class TransitionFeatureVectorConstructionBase(ABC):
    """
    This is the base class for all types that construct a feature vector for a transition
    based on the feature dictionaries of two objects
    """

    @abstractmethod
    def getFeatureNames(self, featureDictObjectA, featureDictObjectB, selectedFeatures) -> list[str]:
        """
        Get a list of feature names describing which features were combined and how.
        e.g. return ['meanIntensityA-meanIntensityB', 'meanIntensityA*meanIntensityB']
        """

    @abstractmethod
    def constructFeatureVector(
        self, featureDictObjectA, featureDictObjectB, selectedFeatures
    ) -> list[npt.NDArray[np.float32]]:
        """
        Set up a feature vector using the selected features of both objects.
        Return a list, not a numpy array!

        The plugin does not need to use all present features, but should not duplicate
        what other plugins already computed.

        e.g. return [featureDictObjectA['meanIntensity']-featureDictObjectB['meanIntensity'],
                    featureDictObjectA['meanIntensity']*featureDictObjectB['meanIntensity']]
        """
