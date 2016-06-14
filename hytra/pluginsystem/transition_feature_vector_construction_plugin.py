from yapsy.IPlugin import IPlugin


class TransitionFeatureVectorConstructionPlugin(IPlugin):
    """
    This is the base class for all plugins that construct a feature vector for a transition
    based on the feature dictionaries of two objects
    """

    def activate(self):
        """
        Activation of plugin could do something, but not needed here
        """
        pass

    def deactivate(self):
        """
        Deactivation of plugin could do something, but not needed here
        """
        pass

    def getFeatureNames(self, featureDictObjectA, featureDictObjectB, selectedFeatures):
        """
        Get a list of feature names describing which features were combined and how.
        e.g. return ['meanIntensityA-meanIntensityB', 'meanIntensityA*meanIntensityB']
        """
        raise NotImplementedError()
        return []

    def constructFeatureVector(self, featureDictObjectA, featureDictObjectB, selectedFeatures):
        """
        Set up a feature vector using the selected features of both objects.
        Return a list, not a numpy array!

        The plugin does not need to use all present features, but should not duplicate
        what other plugins already computed.

        e.g. return [featureDictObjectA['meanIntensity']-featureDictObjectB['meanIntensity'],
                    featureDictObjectA['meanIntensity']*featureDictObjectB['meanIntensity']]
        """
        raise NotImplementedError()
        return []