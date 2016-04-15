from yapsy.PluginManager import PluginManager
import logging
from object_feature_computation_plugin import ObjectFeatureComputationPlugin
from transition_feature_vector_construction_plugin import TransitionFeatureVectorConstructionPlugin
from image_provider_plugin import ImageProviderPlugin
from feature_serializer_plugin import FeatureSerializerPlugin
from merger_resolver_plugin import MergerResolverPlugin


class TrackingPluginManager():
    """
    Our plugin manager that handles the types of plugins known in this pipeline
    """
    def __init__(self, pluginPaths=['plugins'], verbose=False):
        """
        Create the plugin manager that looks inside the specified `pluginPaths` (recursively),
        and if `verbose=True` then the [yapsy](http://yapsy.sourceforge.net/) plugin backend 
        will also show errors that occurred while trying to import plugins (useful for debugging).
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
            "ImageProvider": ImageProviderPlugin,
            "FeatureSerializer": FeatureSerializerPlugin,
            "MergerResolver": MergerResolverPlugin,
        })
        if verbose:
            logging.getLogger('yapsy').setLevel(logging.DEBUG)
        else:
            logging.getLogger('yapsy').setLevel(logging.CRITICAL)

        self._yapsyPluginManager.collectPlugins()
        self.chosen_data_provider = "LocalImageLoader"
        self.chosen_feature_serializer = "LocalFeatureSerializer"
        self.chosen_merger_resolver = 'GMM'

    def _applyToAllPluginsOfCategory(self, func, category):
        ''' helper function to apply `func` to all plugins of the given `category` and hide all yapsy stuff '''
        for pluginInfo in self._yapsyPluginManager.getPluginsOfCategory(category):
            p = pluginInfo.plugin_object
            func(p)

    def _getPluginOfCategory(self, name, category):
        ''' 
        helper function to access a certain plugin by `name` from a `category`. 
        
        **returns** the plugin or throws a `KeyError` 
        '''
        pluginDict = dict((pluginInfo.name, pluginInfo.plugin_object) 
            for pluginInfo in self._yapsyPluginManager.getPluginsOfCategory(category))
        return pluginDict[name]

    def applyObjectFeatureComputationPlugins(self, ndims, rawImage, labelImage, frameNumber, rawFilename):
        """
        computes the features of all plugins and returns a list of dictionaries, as well as a list of
        feature names that should be ignored
        """
        features = []
        featureNamesToIgnore = []
        
        def computeFeatures(plugin):
            if ndims in plugin.worksForDimensions:
                f = plugin.computeFeatures(rawImage, labelImage, frameNumber, rawFilename)
                features.append(f)
                featureNamesToIgnore.extend(plugin.omittedFeatures)
        
        self._applyToAllPluginsOfCategory(computeFeatures, "ObjectFeatureComputation")

        return features, featureNamesToIgnore

    def applyTransitionFeatureVectorConstructionPlugins(self, featureDictObjectA, featureDictObjectB, selectedFeatures):
        """
        constructs a transition feature vector for training/prediction with a random forest from the
        features of the two objects participating in the transition.
        """
        featureVector = []
        def appendFeatures(plugin):
            f = plugin.constructFeatureVector(featureDictObjectA, featureDictObjectB, selectedFeatures)
            featureVector.extend(f)

        self._applyToAllPluginsOfCategory(appendFeatures, "TransitionFeatureVectorConstruction")

        return featureVector

    def getTransitionFeatureNames(self, featureDictObjectA, featureDictObjectB, selectedFeatures):
        """
        returns a verbal description of each feature in the transition feature vector
        """
        featureNames = []
        def collectFeatureNames(plugin):
            f = plugin.getFeatureNames(featureDictObjectA, featureDictObjectB, selectedFeatures)
            featureNames.extend(f)
            
        self._applyToAllPluginsOfCategory(collectFeatureNames, "TransitionFeatureVectorConstruction")
        
        return featureNames

    def setImageProvider(self, imageProviderName):
        ''' set the used image provier plugin name '''
    	self.chosen_data_provider = imageProviderName

    def getImageProvider(self):
        ''' get an instance of the selected image provider plugin '''
        return self._getPluginOfCategory(self.chosen_data_provider, "ImageProvider")

    def setFeatureSerializer(self, featureSerializerName):
        ''' set the used feature serializer plugin name '''
        self.chosen_feature_serializer = featureSerializerName

    def getFeatureSerializer(self):
        ''' get an instance of the selected feature serializer plugin '''
        return self._getPluginOfCategory(self.chosen_feature_serializer, "FeatureSerializer")

    def setMergerResolver(self, mergerResolverName):
        ''' set the used merger resolver plugin name '''
        self.chosen_merger_resolver = mergerResolverName

    def getMergerResolver(self):
        ''' get an instance of the selected merger resolver plugin '''
        return self._getPluginOfCategory(self.chosen_merger_resolver, "MergerResolver")

