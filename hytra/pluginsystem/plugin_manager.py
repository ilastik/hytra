from __future__ import unicode_literals
from yapsy.PluginManager import PluginManager
from yapsy.FilteredPluginManager import FilteredPluginManager
import logging
from hytra.pluginsystem.object_feature_computation_plugin import ObjectFeatureComputationPlugin
from hytra.pluginsystem.transition_feature_vector_construction_plugin import TransitionFeatureVectorConstructionPlugin
from hytra.pluginsystem.image_provider_plugin import ImageProviderPlugin
from hytra.pluginsystem.feature_serializer_plugin import FeatureSerializerPlugin
from hytra.pluginsystem.merger_resolver_plugin import MergerResolverPlugin

class TrackingPluginManager(object):
    """
    Our plugin manager that handles the types of plugins known in this pipeline
    """
    def __init__(self, pluginPaths=['hytra/plugins'], turnOffFeatures=[], verbose=False):
        """
        Create the plugin manager that looks inside the specified `pluginPaths` (recursively),
        and if `verbose=True` then the [yapsy](http://yapsy.sourceforge.net/) plugin backend 
        will also show errors that occurred while trying to import plugins (useful for debugging).
        """
        self._pluginPaths = pluginPaths
        self._turnOffFeatures = turnOffFeatures
        self._verbose = verbose
        
        self._initializeYapsy()

        self.chosen_data_provider = "LocalImageLoader"
        self.chosen_feature_serializer = "LocalFeatureSerializer"
        self.chosen_merger_resolver = 'GMMMergerResolver'

    def __getstate__(self):
        '''
        We define __getstate__ and __setstate__ to exclude the loaded yapsy modules from being pickled.

        See https://docs.python.org/3/library/pickle.html#pickle-state for more details.
        '''
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state['_yapsyPluginManager']
        return state

    def __setstate__(self, state):
        # Restore instance attributes
        self.__dict__.update(state)
        # Restore the yapsy plugins by reading them from scratch
        self._initializeYapsy()

    def _initializeYapsy(self):
        # Build the manager
        self._yapsyPluginManager = PluginManager()
        self._yapsyPluginManager = FilteredPluginManager(self._yapsyPluginManager)
        self._yapsyPluginManager.isPluginOk = lambda x: x.name not in self._turnOffFeatures

        # Tell it the default place(s) where to find plugins
        self._yapsyPluginManager.setPluginPlaces(self._pluginPaths)
        # Define the various categories corresponding to the different
        # kinds of plugins you have defined
        self._yapsyPluginManager.setCategoriesFilter({
            "ObjectFeatureComputation": ObjectFeatureComputationPlugin,
            "TransitionFeatureVectorConstruction": TransitionFeatureVectorConstructionPlugin,
            "ImageProvider": ImageProviderPlugin,
            "FeatureSerializer": FeatureSerializerPlugin,
            "MergerResolver": MergerResolverPlugin,
        })
        if self._verbose:
            logging.getLogger('yapsy').setLevel(logging.DEBUG)
        else:
            logging.getLogger('yapsy').setLevel(logging.CRITICAL)

        self._yapsyPluginManager.collectPlugins()

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

