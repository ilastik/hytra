from pluginsystem import image_provider_plugin
import numpy as np
import h5py


class LocalImageLoader(image_provider_plugin.ImageProviderPlugin):
    """
    Computes the subtraction of features in the feature vector
    """

    shape = None

    def getImageDataAtTimeFrame(self, Resource, PathInResource, timeframe):
        """
        Loads image data from local resource file in hdf5 format.
        PathInResource provides the internal image path 
        Return numpy array of image data at timeframe.
        """
        with h5py.File(Resource, 'r') as rawH5:
            rawImage = rawH5[PathInResource][timeframe, ...]
            return rawImage

    def getLabelImageForFrame(self, Resource, PathInResource, timeframe):
        """
        Loads label image data from local resource file in hdf5 format.
        PathInResource provides the internal image path 
        Return numpy array of image data at timeframe.
        """

        if (self.shape == None):
            self.getImageShape(Resource, PathInResource)

        with h5py.File(Resource, 'r') as h5file:
            labelImage = h5file[
                PathInResource % (timeframe, timeframe + 1, self.shape[0], self.shape[1], self.shape[2])][
                0, ..., 0].squeeze().astype(np.uint32)
            return labelImage

    def getImageShape(self, Resource, PathInResource):
        """
        Derive Image Shape from label image.
        Loads label image data from local resource file in hdf5 format.
        PathInResource provides the internal image path 
        Return list with image dimensions
        """
        with h5py.File(Resource, 'r') as h5file:
            shape = h5file['/'.join(PathInResource.split('/')[:-1])].values()[0].shape[1:4]
            self.shape = shape
            return shape


    def getTimeRange(self, Resource, PathInResource):
        """
        Count Label images to derive the total number of frames
        Loads label image data from local resource file in hdf5 format.
        PathInResource provides the internal image path 
        Return tuple of (first frame, last frame)
        """
        with h5py.File(Resource, 'r') as h5file:
            maxTime = len(h5file['/'.join(PathInResource.split('/')[:-1])].keys())
            return (0,maxTime)