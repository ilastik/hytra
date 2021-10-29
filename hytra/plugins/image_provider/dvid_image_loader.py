from hytra.pluginsystem import image_provider_plugin
import numpy as np
from libdvid import DVIDNodeService

try:
    import json_tricks as json
except ImportError:
    import json


class DvidImageLoader(image_provider_plugin.ImageProviderPlugin):
    """
    Computes the subtraction of features in the feature vector
    """

    shape = None

    def _getRawImageName(self, timeframe):
        return "raw-" + str(timeframe)

    def _getSegmentationName(self, timeframe):
        return "seg-" + str(timeframe)

    def getImageDataAtTimeFrame(self, Resource, PathInResource, axes, timeframe):
        """
        Loads image data from local resource file in hdf5 format.
        PathInResource provides the internal image path
        Return numpy array of image data at timeframe.
        """
        node_service = DVIDNodeService(Resource, PathInResource)

        if self.shape == None:
            self.getImageShape(Resource, PathInResource)

        raw_frame = node_service.get_gray3D(self._getRawImageName(timeframe), tuple(self.shape), (0, 0, 0))
        return raw_frame

    def getLabelImageForFrame(self, Resource, PathInResource, timeframe):
        """
        Loads label image data from local resource file in hdf5 format.
        PathInResource provides the internal image path
        Return numpy array of image data at timeframe.
        """

        if self.shape == None:
            self.getImageShape(Resource, PathInResource)

        node_service = DVIDNodeService(Resource, PathInResource)
        seg_frame = np.array(
            node_service.get_labels3D(self._getSegmentationName(timeframe), tuple(self.shape), (0, 0, 0))
        ).astype(np.uint32)
        return seg_frame

    def getImageShape(self, Resource, PathInResource):
        """
        Derive Image Shape from label image.
        Loads label image data from local resource file in hdf5 format.
        PathInResource provides the internal image path
        Return list with image dimensions
        """

        node_service = DVIDNodeService(Resource, PathInResource)
        config = json.loads(node_service.get("config", "imageInfo"))
        self.shape = config["shape"]
        return self.shape

    def getTimeRange(self, Resource, PathInResource):
        """
        Count Label images to derive the total number of frames
        Loads label image data from local resource file in hdf5 format.
        PathInResource provides the internal image path
        Return tuple of (first frame, last frame)
        """
        node_service = DVIDNodeService(Resource, PathInResource)
        config = json.loads(node_service.get("config", "imageInfo"))
        return config["time_range"]
