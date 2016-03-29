from yapsy.PluginManager import PluginManager
import logging
from object_feature_computation_plugin import ObjectFeatureComputationPlugin

class TrackingPluginManager():
    """
    Our plugin manager that handles the types of plugins known in this pipeline
    """
    def __init__(self, pluginPaths=['plugins'], verbose=False):
        """
        """
        # Build the manager
        self._yapsyPluginManager = PluginManager()
        # Tell it the default place(s) where to find plugins
        self._yapsyPluginManager.setPluginPlaces(pluginPaths)
        # Define the various categories corresponding to the different
        # kinds of plugins you have defined
        self._yapsyPluginManager.setCategoriesFilter({
            "ObjectFeatureComputation": ObjectFeatureComputationPlugin
        })
        if verbose:
            logging.getLogger('yapsy').setLevel(logging.DEBUG)

        self._yapsyPluginManager.collectPlugins()

    def applyObjectFeatureComputationPlugins(self, ndims, rawImage, labelImage, frameNumber):
        """
        computes the features of all plugins and returns a list of dictionaries, as well as a list of
        feature names that should be ignored
        """
        features = []
        featureNamesToIgnore = []
        for pluginInfo in self._yapsyPluginManager.getPluginsOfCategory("ObjectFeatureComputation"):
            p = pluginInfo.plugin_object
            if ndims in p.worksForDimensions:
                f = p.computeFeatures(rawImage, labelImage, frameNumber)
                features.append(f)
                featureNamesToIgnore.extend(p.omittedFeatures)
        return features, featureNamesToIgnore

