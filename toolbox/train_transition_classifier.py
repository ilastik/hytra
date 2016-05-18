from compiler.ast import flatten
import os
import logging
import glob
import vigra
from vigra import numpy as np
import h5py
from sklearn.neighbors import KDTree
from pluginsystem.plugin_manager import TrackingPluginManager

logger = logging.getLogger('TransitionClassifier')
logger.setLevel(logging.DEBUG)

np.seterr(all='raise')

# read in 'n2-n1' of images
def read_in_images(n1, n2, files):
    gt_labelimage = [vigra.impex.readHDF5(f, 'segmentation/labels') for f in files[n1:n2]]
    logger.info("Found segmentation of shape {}".format(gt_labelimage[0].shape))
    return gt_labelimage


# compute features from input data and return them
def compute_features(raw_image, labeled_image, n1, n2, pluginManager, filepath):
    # perhaps there is an elegant way to get into the RegionFeatureAccumulator.
    # For now, the new feature are a separate vector
    allFeat = []
    for i in range(0, n2 - n1):
        if len(labeled_image[i].shape) < len(raw_image.shape) - 1:
            # this was probably a missing channel axis, thus adding one at the end
            labeled_image = np.expand_dims(labeled_image, axis=-1)

        # features = vigra.analysis.extractRegionFeatures(raw_image[..., i, 0].astype('float32'),
        #                                                 labeled_image[i][..., 0], ignoreLabel=0)
        # adapt to this weird format of the 3D labeled image (60,487,1,518) on 3D Fluo-SIM
        if raw_image[..., i, 0].shape != labeled_image[i][..., 0].shape: 
            labeled_image[i] = np.transpose(labeled_image[i], axes=[3,1,0,2])

        moreFeats, ignoreNames = pluginManager.applyObjectFeatureComputationPlugins(
            len(raw_image.shape)-2, raw_image[..., i, 0], labeled_image[i][..., 0], i, filepath)
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
        for key, value in featuresPerFrame.iteritems():
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

class TransitionClassifier:
    def __init__(self, selectedFeatures, numSamples=None):
        """
        Set up a transition classifier class that makes it easy to add samples, train and store the RF.
        :param selectedFeatures: list of feature names that are supposed to be used
        :param numSamples: if given, the data array for the samples is allocated with the proper dimensions,
                            otherwise it needs to be resized whenever new samples are added.
        """
        self.rf = vigra.learning.RandomForest()
        self.mydata = None
        self.labels = []
        self.selectedFeatures = selectedFeatures
        self._numSamples = numSamples
        self._nextIdx = 0
        # TODO: check whether prediction here and in hypotheses graph script are the same!

    def addSample(self, f1, f2, label, pluginManager):
        # if self.labels == []:
        self.labels.append(label)
        # else:
        #    self.labels = np.concatenate((np.array(self.labels),label)) # for adding batches of features
        features = self.constructSampleFeatureVector(f1, f2, pluginManager)

        if self._numSamples is None:
            # use vstack
            if self.mydata is None:
                self.mydata = features
            else:
                self.mydata = np.vstack((self.mydata, features))
        else:
            # allocate full array once, then fill in row by row
            if self.mydata is None:
                self.mydata = np.zeros((self._numSamples, features.shape[0]))

            assert(self._nextIdx < self._numSamples)
            self.mydata[self._nextIdx, :] = features
            self._nextIdx += 1

    def constructSampleFeatureVector(self, f1, f2, pluginManager):
        featVec = pluginManager.applyTransitionFeatureVectorConstructionPlugins(f1, f2, self.selectedFeatures)
        return np.array(featVec)

    # adding a comfortable function, where one can easily introduce the data
    def add_allData(self, mydata, labels):
        self.mydata = mydata
        self.labels = labels

    def train(self, withFeatureImportance=False):
        logger.info(
        "Training classifier from {} positive and {} negative labels".format(
            np.count_nonzero(np.asarray(self.labels)), len(self.labels) - np.count_nonzero(np.asarray(self.labels))))
        logger.info("Training classifier from a feature vector of length {}".format(self.mydata.shape))

        if withFeatureImportance:
            oob, featImportance = self.rf.learnRFWithFeatureSelection(
                self.mydata.astype("float32"),
                (np.asarray(self.labels)).astype("uint32").reshape(-1, 1))
            logger.debug("RF feature importance: {}".format(featImportance))
            # logger.debug('Feature names: {}'.format(self.featureNames))
        else:
            oob = self.rf.learnRF(
                self.mydata.astype("float32"),
                (np.asarray(self.labels)).astype("uint32").reshape(-1, 1))
        logger.info("RF trained with OOB Error {}".format(oob))

    def predictSample(self, test_data=None, f1=None, f2=None):
        if test_data is not None:
            return self.rf.predictLabels(test_data.astype('float32'))
        else:
            data = self.constructSampleFeatureVector(f1, f2)
            if len(data.shape) < 2:
                data = np.expand_dims(data, axis=0)
            return self.rf.predictLabels(data.astype('float32'))

    def predictProbabilities(self, test_data=None, f1=None, f2=None):
        if test_data is not None:
            return self.rf.predictProbabilities(test_data.astype('float32'))
        else:
            data = self.constructSampleFeatureVector(f1, f2)
            print(data)
            if len(data.shape) < 2:
                data = np.expand_dims(data, axis=0)
            return self.rf.predictProbabilities(data.astype('float32'))

    def predictLabels(self, test_data, threshold=0.5):
        prob = self.rf.predictProbabilities(test_data.astype('float32'))
        res = np.copy(prob)
        for i in range(0, len(prob)):
            if prob[i][1] >= threshold:
                res[i] = 1.
            else:
                res[i] = 0
        return np.delete(res, 0, 1)

    def writeRF(self, outputFilename):
        self.rf.writeHDF5(outputFilename, pathInFile='/ClassifierForests/Forest0000')

        # write selected features
        with h5py.File(outputFilename, 'r+') as f:
            featureNamesH5 = f.create_group('SelectedFeatures')
            featureNamesH5 = featureNamesH5.create_group('Standard Object Features')
            for feature in self.selectedFeatures:
                featureNamesH5.create_group(feature)


if __name__ == '__main__':
    import configargparse as argparse

    parser = argparse.ArgumentParser(description="trainRF",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', is_config_file=True, help='config file path')
    parser.add_argument("--groundtruth", dest='filepath', type=str, nargs='+',
                        help="read ground truth from this folder. Can be also a list of paths, to train from more datasets.", metavar="FILE")
    parser.add_argument("--raw-data-file", dest='rawimage_filename', type=str, nargs='+',
                        help="filepath+name of the raw image. Can be a list of paths, to train from more datasets.", metavar="FILE")
    parser.add_argument("--raw-data-path", dest='rawimage_h5_path', type=str,
                        help="Path inside the rawimage HDF5 file", default='volume/data')
    parser.add_argument("--init-frame", default=0, type=int, dest='initFrame',
                        help="where to begin reading the frames")
    parser.add_argument("--end-frame", default=-1, type=int, dest='endFrame',
                        help="where to end frames")
    parser.add_argument("--transition-classifier-file", dest='outputFilename', type=str,
                        help="save RF into file", metavar="FILE")
    parser.add_argument("--filepattern", dest='filepattern', type=str, nargs='+', default=['0*.h5'],
                        help="File pattern of the ground truth files. Can be also a list of paths, to train from more datasets.")
    parser.add_argument("--time-axis-index", dest='time_axis_index', default=2, type=int,
                        help="Zero-based index of the time axis in your raw data. E.g. if it has shape (x,t,y,c) "
                             "this value is 1. Set to -1 to disable any changes. Expected axis order is x,y,(z),t,c")
    parser.add_argument("--verbose", dest='verbose', action='store_true', default=False)

    args, unknown = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO)
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    logging.debug("Ignoring unknown parameters: {}".format(unknown))
    
    assert len(args.rawimage_filename) == len(args.filepattern) == len(args.filepath)
    
    # read raw image
    numSamples = 0
    mlabels = None

    for dataset in range(len(args.rawimage_filename)):
        rawimage_filename = args.rawimage_filename[dataset]
        with h5py.File(rawimage_filename, 'r') as h5raw:
            rawimage = h5raw[args.rawimage_h5_path].value

        # transform such that the order is the following: X,Y,(Z),T, C
        if args.time_axis_index != -1:
            rawimage = np.rollaxis(rawimage, args.time_axis_index, -1)

            # in order to fix shape mismatch between rawimage.shape == (495, 534)
            # and labelimage.shape == (534, 495)
            # rawimage = np.swapaxes(rawimage, 0, 1)
    
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
        trackingPluginManager = TrackingPluginManager()
        features = compute_features(rawimage,
                                        read_in_images(initFrame, endFrame, files),
                                        initFrame,
                                        endFrame,
                                        trackingPluginManager,
                                        rawimage_filename)
        logger.info('Done computing features from dataset {}'.format(dataset))

        selectedFeatures = find_features_without_NaNs(features)
        pos_labels = read_positiveLabels(initFrame, endFrame, files)
        neg_labels = negativeLabels(features, pos_labels)
        numSamples += 2 * sum([len(l) for l in pos_labels]) + sum([len(l) for l in neg_labels])
        logger.info('Done extracting {} samples'.format(numSamples))
        TC = TransitionClassifier(selectedFeatures, numSamples)
        if dataset>0:
            TC.labels = mlabels # restore labels overwritten by constructor

        # compute featuresA for each object A from the feature matrix from Vigra
        def compute_ObjFeatures(features, obj):
            dict = {}
            for key in features:
                if key == "Global<Maximum >" or key == "Global<Minimum >":  # this ones have only one element
                    dict[key] = features[key]
                else:
                    dict[key] = features[key][obj]
            return dict


        for k in range(0, len(features) - 1):
            for i in pos_labels[k]:
                # positive
                logger.debug("Adding positive sample {} from pos {} to {}".format(i,
                              features[k]['RegionCenter'][i[0]], features[k + 1]['RegionCenter'][i[1]]))
                TC.addSample(compute_ObjFeatures(
                    features[k], i[0]), compute_ObjFeatures(features[k + 1], i[1]), 1, trackingPluginManager)
                TC.addSample(compute_ObjFeatures(
                    features[k + 1], i[1]), compute_ObjFeatures(features[k], i[0]), 1, trackingPluginManager)
            for i in neg_labels[k]:
                # negative
                logger.debug("Adding negative sample {} from pos {} to {}".format(i,
                              features[k]['RegionCenter'][i[0]], features[k + 1]['RegionCenter'][i[1]]))
                TC.addSample(compute_ObjFeatures(
                    features[k], i[0]), compute_ObjFeatures(features[k + 1], i[1]), 0, trackingPluginManager)
        mlabels =TC.labels

    logger.info('Done adding samples to RF. Beginning training...')
    TC.train()
    logger.info('Done training RF')

    srcObject = compute_ObjFeatures(features[0], 1)
    destObject = compute_ObjFeatures(features[1], 2)

    # delete file before writing
    if os.path.exists(args.outputFilename):
        os.remove(args.outputFilename)
    TC.writeRF(args.outputFilename)  # writes learned RF to disk
