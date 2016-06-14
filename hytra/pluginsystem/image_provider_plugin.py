from yapsy.IPlugin import IPlugin


class ImageProviderPlugin(IPlugin):
    """
    This is the base class for all plugins that load images from a given location
    """

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


    def getImageDataAtTimeFrame(self, Resource, PathInResource, timeframe):
        """
        Loads image data from location.
        Return one numpy array.
        """
        raise NotImplementedError()
        return []

    def getLabelImageForFrame(self, Resource, PathInResource, timeframe):
        """
        Get the label image(volume) of one time frame
        """
        raise NotImplementedError()
        return []

    def getImageShape(self, Resource, PathInResource):
        """
        extract the shape from the labelimage
        """
        raise NotImplementedError()
        return []

    def getTimeRange(self, Resource, PathInResource):
        """
        extract the time range by counting labelimages
        """
        raise NotImplementedError()
        return []

    def exportLabelImage(self, labelimage, timeframe, Resource, PathInResource):
        """
        export labelimage of timeframe
        """
        raise NotImplementedError()
        return []