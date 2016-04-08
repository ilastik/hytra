from yapsy.PluginManager import PluginManager
import logging
from object_feature_computation_plugin import ObjectFeatureComputationPlugin
from transition_feature_vector_construction_plugin import TransitionFeatureVectorConstructionPlugin
from image_provider_plugin import ImageProviderPlugin

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
            "ObjectFeatureComputation": ObjectFeatureComputationPlugin,
            "TransitionFeatureVectorConstruction": TransitionFeatureVectorConstructionPlugin,
            "ImageProvider": ImageProviderPlugin
        })
        if verbose:
            logging.getLogger('yapsy').setLevel(logging.DEBUG)

        self._yapsyPluginManager.collectPlugins()
        self.chosen_data_provider = "LocalImageLoader"

    def applyObjectFeatureComputationPlugins(self, ndims, rawImage, labelImage, frameNumber, rawFilename):
        """
        computes the features of all plugins and returns a list of dictionaries, as well as a list of
        feature names that should be ignored
        """
        features = []
        featureNamesToIgnore = []
        for pluginInfo in self._yapsyPluginManager.getPluginsOfCategory("ObjectFeatureComputation"):
            p = pluginInfo.plugin_object
            if ndims in p.worksForDimensions:
                f = p.computeFeatures(rawImage, labelImage, frameNumber, rawFilename)
                features.append(f)
                featureNamesToIgnore.extend(p.omittedFeatures)
        return features, featureNamesToIgnore

    def applyTransitionFeatureVectorConstructionPlugins(self, featureDictObjectA, featureDictObjectB, selectedFeatures):
        """
        constructs a transition feature vector for training/prediction with a random forest from the
        features of the two objects participating in the transition.
        """
        featureVector = []
        for pluginInfo in self._yapsyPluginManager.getPluginsOfCategory("TransitionFeatureVectorConstruction"):
            p = pluginInfo.plugin_object
            f = p.constructFeatureVector(featureDictObjectA, featureDictObjectB, selectedFeatures)
            featureVector.extend(f)
        return featureVector

    def getTransitionFeatureNames(self, featureDictObjectA, featureDictObjectB, selectedFeatures):
        """
        returns a verbal description of each feature in the transition feature vector
        """
        featureNames = []
        for pluginInfo in self._yapsyPluginManager.getPluginsOfCategory("TransitionFeatureVectorConstruction"):
            p = pluginInfo.plugin_object
            f = p.getFeatureNames(featureDictObjectA, featureDictObjectB, selectedFeatures)
            featureNames.extend(f)
        return featureNames

    def setImageProvider(self, ImageProviderName):
    	self.chosen_data_provider = ImageProviderName

    def getImageProvider(self):
    	PrividerDict = dict((pluginInfo.name, pluginInfo.plugin_object) for pluginInfo in self._yapsyPluginManager.getPluginsOfCategory("ImageProvider"))
    	return PrividerDict[self.chosen_data_provider]