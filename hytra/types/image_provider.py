from abc import ABC, abstractmethod


class ImageProviderPlugin(ABC):
    """
    This is the base class for all plugins that load images from a given location
    """

    @abstractmethod
    def getImageDataAtTimeFrame(self, Resource, PathInResource, axes, timeframe):
        """
        Loads image data from location.
        Return one numpy array.
        """

    @abstractmethod
    def getLabelImageForFrame(self, Resource, PathInResource, timeframe):
        """
        Get the label image(volume) of one time frame
        """

    @abstractmethod
    def getImageShape(self, Resource, PathInResource):
        """
        extract the shape from the labelimage
        """

    @abstractmethod
    def getTimeRange(self, Resource, PathInResource):
        """
        extract the time range by counting labelimages
        """

    @abstractmethod
    def exportLabelImage(self, labelimage, timeframe, Resource, PathInResource):
        """
        export labelimage of timeframe
        """
