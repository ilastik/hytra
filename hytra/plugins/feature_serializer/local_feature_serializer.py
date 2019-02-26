from hytra.pluginsystem import feature_serializer_plugin
import numpy as np


class LoadFeatureSerializer(feature_serializer_plugin.FeatureSerializerPlugin):
    """
    serializes features into local dict
    """

    def storeFeaturesForFrame(self, features, timeframe):
        """
        Stores feature data
        """
        assert self.features_per_frame is not None
        assert isinstance(self.features_per_frame, dict)
        self.features_per_frame[timeframe] = features

    def loadFeaturesForFrame(self, features, timeframe):
        """
        loads feature data
        """
        assert self.features_per_frame is not None
        assert isinstance(self.features_per_frame, dict)
        return self.features_per_frame[timeframe]
