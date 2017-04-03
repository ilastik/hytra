from __future__ import print_function, absolute_import, nested_scopes, generators, division, with_statement, unicode_literals

from yapsy.IPlugin import IPlugin


class FeatureSerializerPlugin(IPlugin):
    """
    This is the base class for all plugins that load/store the features to/from different locations
    """

    server_address = None
    ''' Address of the dvid server (only used by the dvid serializer plugin) '''

    uuid = None
    ''' Address of the dataset uuid on the dvid server (only used by the dvid serializer plugin) '''

    features_per_frame = None
    ''' dictionary of features per frame (only used by local serializer plugin) '''

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

    def storeFeaturesForFrame(self, features, timeframe):
        """
        Stores feature data
        """
        raise NotImplementedError()
        return []

    def loadFeaturesForFrame(self, features, timeframe):
        """
        loads feature data
        """
        raise NotImplementedError()
        return []