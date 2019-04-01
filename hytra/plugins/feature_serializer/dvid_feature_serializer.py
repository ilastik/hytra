from hytra.pluginsystem import feature_serializer_plugin
from libdvid import DVIDNodeService

try:
    import json_tricks as json
except ImportError:
    import json


class DvidFeatureSerializer(feature_serializer_plugin.FeatureSerializerPlugin):
    """
    serializes features to dvid
    """

    keyvalue_store = "features"

    def storeFeaturesForFrame(self, features, timeframe):
        """
        Stores feature data
        """
        assert self.server_address is not None
        assert self.uuid is not None
        node_service = DVIDNodeService(self.server_address, self.uuid)
        node_service.create_keyvalue(self.keyvalue_store)
        node_service.put(
            self.keyvalue_store, "frame-{}".format(timeframe), json.dumps(features)
        )

    def loadFeaturesForFrame(self, features, timeframe):
        """
        loads feature data
        """
        assert self.server_address is not None
        assert self.uuid is not None
        node_service = DVIDNodeService(self.server_address, self.uuid)
        node_service.create_keyvalue(self.keyvalue_store)
        return json.loads(
            node_service.get(self.keyvalue_store, "frame-{}".format(timeframe))
        )
