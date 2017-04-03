from hytra.pluginsystem import image_provider_plugin
import hytra.util.axesconversion
import numpy as np
import h5py
import logging

class LocalImageLoader(image_provider_plugin.ImageProviderPlugin):
    """
    Computes the subtraction of features in the feature vector
    """

    shape = None

    def getImageDataAtTimeFrame(self, Resource, PathInResource, axes, timeframe):
        """
        Loads image data from local resource file in hdf5 format.
        PathInResource provides the internal image path 
        Return numpy array of image data at timeframe.
        """
        logging.getLogger("LocalImageLoader").debug("opening {}".format(Resource))
        with h5py.File(Resource, 'r') as rawH5:
            logging.getLogger("LocalImageLoader").debug("PathInResource {}".format(timeframe))
            rawImage = rawH5[PathInResource][hytra.util.axesconversion.getFrameSlicing(axes, timeframe)]
            remainingAxes = axes.replace('t', '')
            rawImage = hytra.util.axesconversion.adjustOrder(rawImage, remainingAxes).squeeze()
            return rawImage

    def getLabelImageForFrame(self, Resource, PathInResource, timeframe):
        """
        Loads label image data from local resource file in hdf5 format.
        PathInResource provides the internal image path, which -- depending on the ilastik version --
        can be a LabelImage path where the shape is encoded in the individual timeframe-blocks (5x %d in the string),
        or a `LabelImage_v2` path which contains blocks that have a `blockSlice` attribute telling us which part of the data they contain.

        Return numpy array of image data at timeframe.
        """

        if (self.shape == None):
            self.getImageShape(Resource, PathInResource)

        with h5py.File(Resource, 'r') as h5file:
            if PathInResource.count('%') == 5 and not 'LabelImage_v2' in PathInResource:
                internalPath = PathInResource % (timeframe, timeframe + 1, self.shape[0], self.shape[1], self.shape[2])
                logging.getLogger("LocalImageLoader").debug("Opening label image at {}".format(internalPath))
                labelImage = h5file[internalPath][0, ..., 0]
            elif 'LabelImage_v2' in PathInResource:
                # loop though all blocks in h5 file and read the blockshape, and figure out whether this frame is in there.
                labelImage = None
                for block in h5file[PathInResource].values():
                    assert 'blockSlice' in block.attrs
                    blockSlice = block.attrs['blockSlice']
                    bs = blockSlice[1:-1]
                    roi = [(int(r.split(':')[0]), int(r.split(':')[1])) for r in bs.split(',')]
                    if timeframe in range(roi[0][0], roi[0][1]):
                        timeStart = timeframe - roi[0][0]
                        # WARNING we assume that every block captures the full image extent or more timeframes, 
                        # which might not always be true (are 3D frames separated into multiple blocks?)
                        labelImage = block[timeStart:timeStart+1, ..., 0]
                        break
                assert labelImage is not None
            else:
                raise ValueError("Invalid PathInResource: {}".format(PathInResource))
            return labelImage.squeeze().astype(np.uint32)

    def getImageShape(self, Resource, PathInResource):
        """
        Derive Image Shape from label image.
        Loads label image data from local resource file in hdf5 format.
        PathInResource provides the internal image path 
        Return list with image dimensions.

        Works with both `PathInResource` styles: LabelImage and LabelImage_v2
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
            return (0, maxTime)

    def exportLabelImage(self, labelimage, timeframe, Resource, PathInResource):
        """
        export labelimage of timeframe
        """
        with h5py.File(Resource, 'r+') as h5file:
            if PathInResource.count('%') == 5 and not 'LabelImage_v2' in PathInResource:
                internalPath = PathInResource % (timeframe, timeframe + 1, self.shape[0], self.shape[1], self.shape[2])
                blockSlice = None
            elif 'LabelImage_v2' in PathInResource:
                # loop though all blocks in h5 file and read the blockshape, and figure out whether this frame is in there.
                labelImage = None
                if PathInResource in h5file and len(h5file[PathInResource].keys()) > 0:
                    lastBlock = sorted(h5file[PathInResource].keys())[-1]
                    lastBlockNr = int(lastBlock.replace('block', ''))
                else:
                    lastBlockNr = -1
                newBlockName = "block{:04d}".format(lastBlockNr+1)
                blockSlice = "[{}:{},{}:{},{}:{},{}:{},{}:{}]".format(timeframe, timeframe+1, 0, self.shape[0], 0, self.shape[1], 0, self.shape[2], 0, 1)
                internalPath = '/'.join([PathInResource, newBlockName])
            else:
                raise ValueError("Invalid PathInResource: {}".format(PathInResource))
            
            if len(labelimage.shape) == 3:
                h5file.create_dataset(internalPath, data=labelimage[np.newaxis, :, :, :, np.newaxis], dtype='u2', compression='gzip')
            elif len(labelimage.shape) == 2:
                h5file.create_dataset(internalPath, data=labelimage[np.newaxis, :, :, np.newaxis, np.newaxis], dtype='u2', compression='gzip')
            else:
                raise NotImplementedError()
            
            if blockSlice is not None:
                h5file[internalPath].attrs['blockSlice'] = blockSlice
