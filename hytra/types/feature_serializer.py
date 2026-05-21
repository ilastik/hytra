from abc import ABC, abstractmethod


class FeatureSerializerPlugin(ABC):
    """
    This is the base class for all plugins that load/store the features to/from different locations
    """

    server_address = None
    """ Address of the dvid server (only used by the dvid serializer plugin) """

    uuid = None
    """ Address of the dataset uuid on the dvid server (only used by the dvid serializer plugin) """

    features_per_frame = None
    """ dictionary of features per frame (only used by local serializer plugin) """

    @abstractmethod
    def storeFeaturesForFrame(self, features, timeframe):
        """
        Stores feature data
        """

    @abstractmethod
    def loadFeaturesForFrame(self, features, timeframe):
        """
        loads feature data
        """
