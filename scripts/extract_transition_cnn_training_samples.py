# pythonpath modification to make hytra and empryonic available 
# for import without requiring it to be installed
from __future__ import print_function, absolute_import, nested_scopes, generators, division, with_statement, unicode_literals
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
# standard imports
from compiler.ast import flatten
import logging
import glob
import vigra
from vigra import numpy as np
import h5py
from sklearn.neighbors import KDTree
from hytra.pluginsystem.plugin_manager import TrackingPluginManager
import hytra.util.axesconversion

logger = logging.getLogger('TransitionClassifier')
logger.setLevel(logging.DEBUG)

np.seterr(all='raise')

# read in 'n2-n1' of images
def read_in_images(n1, n2, files, axes):
    gt_labelimage = [vigra.impex.readHDF5(f, 'segmentation/labels') for f in files[n1:n2]]
    gt_labelimage = [hytra.util.axesconversion.adjustOrder(img, axes, 'xyzc') for img in gt_labelimage]
    logger.info("Found segmentation of shape {}".format(gt_labelimage[0].shape))
    return gt_labelimage

# compute features from input data and return them
def compute_features(raw_image, labeled_image, n1, n2, pluginManager, filepath):
    # perhaps there is an elegant way to get into the RegionFeatureAccumulator.
    # For now, the new feature are a separate vector
    allFeat = []
    for i in range(0, n2 - n1):
        moreFeats, _ = pluginManager.applyObjectFeatureComputationPlugins(
            len(raw_image.squeeze().shape)-1, raw_image[..., i, 0], labeled_image[i][..., 0], i, filepath)
        frameFeatureItems = []
        for f in moreFeats:
            frameFeatureItems = frameFeatureItems + f.items()
        allFeat.append(dict(frameFeatureItems))
    return allFeat

# read in 'n2-n1' of labels
def read_positiveLabels(n1, n2, files):
    gt_moves = [vigra.impex.readHDF5(f, 'tracking/Moves') for f in files[n1+1:n2]]
    return gt_moves

def getValidRegionCentersAndTheirIDs(featureDict, 
                                     countFeatureName='Count', 
                                     regionCenterName='RegionCenter'):
    """
    From the feature dictionary of a certain frame, 
    find all objects with pixel count > 0, and return their
    region centers and ids.
    """
    validObjectMask = featureDict[countFeatureName] > 0
    validObjectMask[0] = False
    
    regionCenters = featureDict[regionCenterName][validObjectMask, :]
    objectIds = list(np.where(validObjectMask)[0])
    return regionCenters, objectIds

def negativeLabels(features, positiveLabels):
    """
    Compute negative labels by finding 3 nearest neighbors in the next frame, and
    filtering out those pairings that are part of the positiveLabels.

    **Returns** a list of lists of pairs of indices, where there are as many inner lists
    as there are pairs of consecutive frames ordered by time, 
    e.g. for frame pairs (0,1), (1,2), ... (n-1,n).
    Each pair in such a list then contains an index into the earlier frame of the pair, 
    and one index into the later frame.
    """
    numFrames = len(features)
    neg_lab = []
    for i in range(1, numFrames):  # for all frames but the first
        logger.debug("Frame {}\n".format(i))
        frameNegLab = []
        # build kdtree for frame i
        centersAtI, objectIdsAtI = getValidRegionCentersAndTheirIDs(features[i])
        kdt = KDTree(centersAtI, metric='euclidean')
        # find k=3 nearest neighbors of each object of frame i-1 in frame i
        centersAtIMinusOne, objectIdsAtIMinusOne = getValidRegionCentersAndTheirIDs(features[i - 1])
        neighb = kdt.query(centersAtIMinusOne, k=3, return_distance=False)
        for j in range(0, neighb.shape[0]):  # for all valid objects in frame i-1
            logger.debug('looking at neighbors of {} at position {}'.format(
                objectIdsAtIMinusOne[j], features[i - 1]['RegionCenter'][objectIdsAtIMinusOne[j], ...]))
            for m in range(0, neighb.shape[1]):  # for all neighbors
                pair = [objectIdsAtIMinusOne[j], objectIdsAtI[neighb[j][m]]]
                if pair not in positiveLabels[i - 1].tolist():
                    # add one because we've removed the first element when creating the KD tree
                    frameNegLab.append(pair)
                    logger.debug("Adding negative example: {} at position {}".format(
                        pair, features[i]['RegionCenter'][objectIdsAtI[neighb[j][m]], ...]))
                else:
                    logger.debug("Discarding negative example {} which is a positive annotation".format(pair))
        neg_lab.append(frameNegLab)

    return neg_lab

def find_features_without_NaNs(features):
    """
    Remove all features from the list of selected features which have NaNs
    """
    selectedFeatures = features[0].keys()
    for featuresPerFrame in features:
        for key, value in featuresPerFrame.items():
            if not isinstance(value, list) and (np.any(np.isnan(value)) or np.any(np.isinf(value))):
                try:
                    selectedFeatures.remove(key)
                except:
                    pass  # has already been deleted
    forbidden = ["Global<Maximum >", "Global<Minimum >", 'Histogram', 'Polygon', 'Defect Center',
                 'Center', 'Input Center', 'Weighted<RegionCenter>']
    for f in forbidden:
        if f in selectedFeatures:
            selectedFeatures.remove(f)

    selectedFeatures.sort()
    return selectedFeatures

class BoundingBox(object):
    origin = None
    shape = None

    def __init__(self, origin, shape):
        assert len(origin) == len(shape), "BoundingBox.__init__: origin and shape must have same shape!"
        self.origin = np.array(origin)
        self.shape = np.array(shape)

        if len(self.origin) == 2:
            self.origin = np.hstack([self.origin, [0]])
            self.shape = np.hstack([self.shape, [1]])
        elif len(self.origin) != 3:
            raise ValueError("Dimensionality of bounding box must be 2 or 3!")

    @classmethod
    def surroundBothObjects(cls, coordMinA, coordMaxA, coordMinB, coordMaxB):
        '''
        Takes the bounding boxes of two objects as min and max coordinate lists,
        finds the smalles enclosing cuboid, and returns a BoundingBox instance configured with that.
        '''
        origin = np.array([min(coordMinA[i], coordMinB[i]) for i in range(len(coordMinA))])
        shape = np.array([max(coordMaxA[i], coordMaxB[i]) for i in range(len(coordMinA))]) - origin
        return cls(origin, shape)

    def __repr__(self):
        return "BoundingBox<origin={}, shape={}>".format(self.origin, self.shape)

    def expandToRequestedSize(self, maxImageShape, size):
        '''
        make BB bigger but respect image size
        '''
        assert len(maxImageShape) == len(self.origin)
        for dimIdx in range(len(self.origin)):
            requiredPadding = size - self.shape[dimIdx]
            halfPadding = int(np.floor(requiredPadding / 2))
            spaceAfter = maxImageShape[dimIdx] - self.origin[dimIdx] - self.shape[dimIdx]
            if self.origin[dimIdx] < halfPadding:
                paddingBefore = self.origin[dimIdx]
                remainingPadding = requiredPadding - paddingBefore
                if spaceAfter < remainingPadding:
                    paddingAfter = spaceAfter
                else:
                    paddingAfter = remainingPadding
            elif spaceAfter < halfPadding + 1:
                paddingAfter = spaceAfter
                remainingPadding = requiredPadding - paddingAfter
                if self.origin[dimIdx] < remainingPadding:
                    paddingBefore = self.origin[dimIdx]
                else:
                    paddingBefore = remainingPadding
            else:
                paddingBefore = halfPadding
                paddingAfter = requiredPadding - halfPadding

            allowedPadding = paddingBefore + paddingAfter
            self.origin[dimIdx] -= paddingBefore
            self.shape[dimIdx] += allowedPadding
            # self.shape[dimIdx] = min(self.shape[dimIdx] + requiredPadding, maxImageShape[dimIdx] - self.origin[dimIdx])
            assert self.origin[dimIdx] >= 0, "Origin moved past lower bound"
            assert self.origin[dimIdx] + self.shape[dimIdx] <= maxImageShape[dimIdx], "Shape got too big"
        return self


if __name__ == '__main__':
    import configargparse as argparse

    parser = argparse.ArgumentParser(description="Extract training instances for a CNN based transition classifier",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', is_config_file=True, help='config file path')
    parser.add_argument("--groundtruth", dest='filepath', type=str, nargs='+',
                        help="read ground truth from this folder. Can be also a list of paths, to train from more datasets.", metavar="FILE")
    parser.add_argument("--groundtruth-axes", dest='groundtruth_axes', type=str, nargs='+', default=['xyzc'],
                        help="axes ordering of the ground truth segmentations per frame (no t!), e.g. xyzc", metavar="FILE")
    parser.add_argument("--raw-data-file", dest='rawimage_filename', type=str, nargs='+',
                        help="filepath+name of the raw image. Can be a list of paths, to train from more datasets.", metavar="FILE")
    parser.add_argument("--raw-data-path", dest='rawimage_h5_path', type=str,
                        help="Path inside the rawimage HDF5 file", default='volume/data')
    parser.add_argument("--raw-data-axes", dest='rawimage_axes', type=str, nargs='+', default=['txyzc'],
                        help="axes ordering of the raw image, e.g. xyztc. Can be a list of paths, to train from more datasets.", metavar="FILE")
    parser.add_argument("--init-frame", default=0, type=int, dest='initFrame',
                        help="where to begin reading the frames")
    parser.add_argument("--end-frame", default=-1, type=int, dest='endFrame',
                        help="where to end frames")
    parser.add_argument("--out", dest='outputFilename', type=str,
                        help="save training samples into this HDF5 file", metavar="FILE")
    parser.add_argument("--filepattern", dest='filepattern', type=str, nargs='+', default=['0*.h5'],
                        help="File pattern of the ground truth files. Can be also a list of paths, to train from more datasets.")
    parser.add_argument("--verbose", dest='verbose', action='store_true', default=False)
    parser.add_argument('--plugin-paths', dest='pluginPaths', type=str, nargs='+',
                        default=[os.path.abspath('../hytra/plugins')],
                        help='A list of paths to search for plugins for the tracking pipeline.')

    args, unknown = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO)
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    logging.debug("Ignoring unknown parameters: {}".format(unknown))
    
    assert len(args.rawimage_filename) == len(args.rawimage_axes) == len(args.filepattern) == len(args.filepath) == len(args.groundtruth_axes)
    
    # read raw image
    numSamples = 0
    mlabels = None

    for dataset in range(len(args.rawimage_filename)):
        rawimage_filename = args.rawimage_filename[dataset]
        with h5py.File(rawimage_filename, 'r') as h5raw:
            rawimage = h5raw[args.rawimage_h5_path].value

        # transform such that the order is the following: X,Y,(Z),T, C
        rawimage = hytra.util.axesconversion.adjustOrder(rawimage, args.rawimage_axes[dataset], 'xyztc')
        logger.info('Done loading raw data from dataset {} of shape {}'.format(dataset, rawimage.shape))

        # find ground truth files
        # filepath is now a list of filepaths'
        filepath = args.filepath[dataset]
        # filepattern is now a list of filepatterns
        files = glob.glob(os.path.join(filepath, args.filepattern[dataset]))
        files.sort()
        initFrame = args.initFrame
        endFrame = args.endFrame
        if endFrame < 0:
            endFrame += len(files)
    
        # compute features
        trackingPluginManager = TrackingPluginManager(verbose=args.verbose, 
                                                      pluginPaths=args.pluginPaths)
        gt_labelimage = read_in_images(initFrame, endFrame, files, args.groundtruth_axes[dataset])
        features = compute_features(rawimage,
                                    gt_labelimage,
                                    initFrame,
                                    endFrame,
                                    trackingPluginManager,
                                    rawimage_filename)
        logger.info('Done computing features from dataset {}'.format(dataset))

        selectedFeatures = find_features_without_NaNs(features)
        logger.info("Using features: {}".format(selectedFeatures))
        pos_labels = read_positiveLabels(initFrame, endFrame, files)
        neg_labels = negativeLabels(features, pos_labels)
        all_labels = [pos_labels, neg_labels]
        numSamples += 2 * sum([len(l) for l in pos_labels]) + sum([len(l) for l in neg_labels])
        logger.info('Done extracting {} samples'.format(numSamples))

        # find biggest required bounding box:
        maxBoundingBoxSize = 0
        for k in range(0, len(features) - 1):
            for l in [0, 1]: # neg and pos label
                for i in all_labels[l][k]:
                    logger.debug("Adding positive sample {} from pos {} to {}".format(i, features[k]['RegionCenter'][i[0]], features[k + 1]['RegionCenter'][i[1]]))
                    boundingBox = BoundingBox.surroundBothObjects(features[k    ]['Coord<Minimum >'][i[0]], 
                                                                  features[k    ]['Coord<Maximum >'][i[0]],
                                                                  features[k + 1]['Coord<Minimum >'][i[1]], 
                                                                  features[k + 1]['Coord<Maximum >'][i[1]])
                    # logger.info("Found bounding box {} around objects {}-{} and {}-{}".format(boundingBox,
                    #                                               features[k    ]['Coord<Minimum >'][i[0]], 
                    #                                               features[k    ]['Coord<Maximum >'][i[0]],
                    #                                               features[k + 1]['Coord<Minimum >'][i[1]], 
                    #                                               features[k + 1]['Coord<Maximum >'][i[1]]))
                    maxBoundingBoxSize = max(maxBoundingBoxSize, boundingBox.shape.max())

        logger.info("Maximal required bounding box size: {}".format(maxBoundingBoxSize))

        sampleVolumeShape = [numSamples, 4, maxBoundingBoxSize, maxBoundingBoxSize]
        if rawimage.shape[2] == 1:
            sampleVolumeShape += [1]
        else:
            sampleVolumeShape += [maxBoundingBoxSize]

        sampleVolume = np.zeros(sampleVolumeShape, dtype=rawimage.dtype)
        labels = np.zeros(numSamples, dtype=np.uint8)

        # extract crops from src and dest images for all samples
        imageShape = rawimage.shape[:3]
        idx = 0
        for k in range(0, len(features) - 1):
            for l in [0, 1]: # neg and pos label
                for i in all_labels[l][k]:
                    boundingBox = BoundingBox.surroundBothObjects(features[k    ]['Coord<Minimum >'][i[0]], 
                                                                  features[k    ]['Coord<Maximum >'][i[0]],
                                                                  features[k + 1]['Coord<Minimum >'][i[1]], 
                                                                  features[k + 1]['Coord<Maximum >'][i[1]])
                    # logger.info("Bounding Box before expanding: {}".format(boundingBox))
                    # logger.info("Requiring size {}, where image has shape {}".format(maxBoundingBoxSize, imageShape))
                    boundingBox.expandToRequestedSize(imageShape, maxBoundingBoxSize)
                    # logger.info("Bounding Box after expanding: {}".format(boundingBox))
                    sampleVolume[idx, 0, ...] = rawimage[boundingBox.origin[0]:boundingBox.origin[0]+boundingBox.shape[0],
                                                         boundingBox.origin[1]:boundingBox.origin[1]+boundingBox.shape[1],
                                                         boundingBox.origin[2]:boundingBox.origin[2]+boundingBox.shape[2],
                                                         k, 0]
                    sampleVolume[idx, 1, ...] = rawimage[boundingBox.origin[0]:boundingBox.origin[0]+boundingBox.shape[0],
                                                         boundingBox.origin[1]:boundingBox.origin[1]+boundingBox.shape[1],
                                                         boundingBox.origin[2]:boundingBox.origin[2]+boundingBox.shape[2], 
                                                         k + 1, 0]
                    sampleVolume[idx, 2, ...] = gt_labelimage[k][boundingBox.origin[0]:boundingBox.origin[0]+boundingBox.shape[0],
                                                              boundingBox.origin[1]:boundingBox.origin[1]+boundingBox.shape[1],
                                                              boundingBox.origin[2]:boundingBox.origin[2]+boundingBox.shape[2],
                                                              0]
                    sampleVolume[idx, 3, ...] = gt_labelimage[k + 1][boundingBox.origin[0]:boundingBox.origin[0]+boundingBox.shape[0],
                                                                     boundingBox.origin[1]:boundingBox.origin[1]+boundingBox.shape[1],
                                                                     boundingBox.origin[2]:boundingBox.origin[2]+boundingBox.shape[2], 
                                                                     0]
                    labels[idx] = l
                    idx += 1

    logger.info('Done extracting samples, saving to disk')

    if os.path.exists(args.outputFilename):
        logger.info("Removing already existing HDF5 file (because otherwise the h5 file grows!)")
        os.remove(args.outputFilename)

    with h5py.File(args.outputFilename, 'w') as h5out:
        h5out.create_dataset('images', data=sampleVolume, compression='gzip')
        h5out.create_dataset('labels', data=labels)
    logger.info("Done")