from compiler.ast import flatten
import os
import vigra
from vigra import numpy as np
import h5py
from sklearn.neighbors import KDTree
import logging
from pluginsystem.plugin_manager import TrackingPluginManager

logger = logging.getLogger('TransitionClassifier')
logger.setLevel(logging.DEBUG)

np.seterr(all='raise')


# read in 'n2-n1' of images
def read_in_images(n1, n2, filepath, fileFormatString='{:05}.h5'):
    gt_labelimage_filename = []
    for i in range(n1, n2):
        gt_labelimage_filename.append(os.path.join(str(filepath), fileFormatString.format(i)))
    gt_labelimage = [vigra.impex.readHDF5(gt_labelimage_filename[i], 'segmentation/labels') for i in range(0, n2 - n1)]
    return gt_labelimage


# compute features from input data and return them
def compute_features(raw_image, labeled_image, n1, n2, pluginManager):
    # perhaps there is an elegant way to get into the RegionFeatureAccumulator.
    # For now, the new feature are a separate vector
    allFeat = []
    for i in range(0, n2 - n1):
        if len(labeled_image[i].shape) < len(raw_image.shape) - 1:
            # this was probably a missing channel axis, thus adding one at the end
            labeled_image = np.expand_dims(labeled_image, axis=-1)

        # features = vigra.analysis.extractRegionFeatures(raw_image[..., i, 0].astype('float32'),
        #                                                 labeled_image[i][..., 0], ignoreLabel=0)
        moreFeats, ignoreNames = pluginManager.applyObjectFeatureComputationPlugins(len(raw_image.shape) - 2,
                                                raw_image[..., i, 0],
                                                labeled_image[i][..., 0],
                                                i)
        frameFeatureItems = []
        for f in moreFeats:
            frameFeatureItems = frameFeatureItems + f.items()
        allFeat.append(dict(frameFeatureItems))
    return allFeat


# return a feature vector of two objects (f1-f2,f1*f2)
def getFeatures(f1, f2, o1, o2):
    res = []
    res2 = []
    for key in f1:
        if key == "Global<Maximum >" or key == "Global<Minimum >":
            # the global min/max intensity is not interesting
            continue
        elif key == 'RegionCenter':
            res.append(np.linalg.norm(f1[key][o1] - f2[key][o2]))  # difference of features
            res2.append(np.linalg.norm(f1[key][o1] * f2[key][o2]))  # product of features
        elif key == 'Histogram':  # contains only zeros, so trying to see what the prediction is without it
            continue
        elif key == 'Polygon':  # vect has always another length for different objects, so center would be relevant
            continue
        else:
            res.append((f1[key][o1] - f2[key][o2]).tolist())  # prepare for flattening
            res2.append((f1[key][o1] * f2[key][o2]).tolist())  # prepare for flattening
    x = np.asarray(flatten(res))  # flatten
    x2 = np.asarray(flatten(res2))  # flatten
    # x= x[~np.isnan(x)]
    # x2= x2[~np.isnan(x2)] #not getting the nans out YET
    return np.concatenate((x, x2))


# read in 'n2-n1' of labels
def read_positiveLabels(n1, n2, filepath, fileFormatString='{:05}.h5'):
    gt_labels_filename = []
    for i in range(n1 + 1, n2):  # the first one contains no moves data
        gt_labels_filename.append(os.path.join(str(filepath), fileFormatString.format(i)))
    gt_labelimage = [vigra.impex.readHDF5(f, 'tracking/Moves') for f in gt_labels_filename]
    return gt_labelimage


# compute negative labels by nearest neighbor
def negativeLabels(features, positiveLabels):
    numFrames = len(features)
    neg_lab = []
    for i in range(1, numFrames):  # for all frames but the first
        logger.debug("Frame {}\n".format(i))
        frameNegLab = []
        # build kdtree for frame i
        kdt = KDTree(features[i]['RegionCenter'][1:, ...], metric='euclidean')
        # find k=3 nearest neighbors of each object of frame i-1 in frame i
        neighb = kdt.query(features[i - 1]['RegionCenter'][1:, ...], k=3, return_distance=False)
        for j in range(0, neighb.shape[0]):  # for all objects in frame i-1
            logger.debug('looking at neighbors of {} at position {}'.format(j + 1, features[i - 1]['RegionCenter'][j + 1, ...]))
            for m in range(0, neighb.shape[1]):  # for all neighbors
                pair = [j + 1, neighb[j][m] + 1]
                if pair not in positiveLabels[i - 1].tolist():
                    # add one because we've removed the first element when creating the KD tree
                    frameNegLab.append(pair)
                    logger.debug("Adding negative example: {} at position {}".format(pair, features[i]['RegionCenter'][neighb[j][m] + 1, ...]))
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
        self.featureNames = []
        self._numSamples = numSamples
        self._nextIdx = 0
        # TODO: check whether prediction here and in hypotheses graph script are the same!

    def addSample(self, f1, f2, label):
        # if self.labels == []:
        self.labels.append(label)
        # else:
        #    self.labels = np.concatenate((np.array(self.labels),label)) # for adding batches of features
        features = self.constructSampleFeatureVector(f1, f2)

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

    def constructSampleFeatureVector(self, f1, f2):
        res = []
        res2 = []
        names = []
        names2 = []
        for key in self.selectedFeatures:
            if key == "Global<Maximum >" or key == "Global<Minimum >":
                # the global min/max intensity is not interesting
                continue
            elif key == 'RegionCenter':
                res.append(np.linalg.norm(f1[key] - f2[key]))  # difference of features
                res2.append(np.linalg.norm(f1[key] * f2[key]))  # product of features
                names.append('distance')
                names2.append('distance')
            elif key == 'Histogram':  # contains only zeros, so trying to see what the prediction is without it
                continue
            elif key == 'Polygon':  # vect has always another length for different objects, so center would be relevant
                continue
            else:
                if not isinstance(f1[key], np.ndarray):
                    res.append(float(f1[key]) - float(f2[key]))  # prepare for flattening
                    res2.append(float(f1[key]) * float(f2[key]))  # prepare for flattening
                    names.append(key)
                    names2.append(key)
                else:
                    res.append((f1[key] - f2[key]).tolist())  # prepare for flattening
                    res2.append((f1[key] * f2[key]).tolist())  # prepare for flattening
                    for _ in range((f1[key] - f2[key]).size):
                        names.append(key)
                        names2.append(key)
        x = np.asarray(flatten(res))  # flatten
        x2 = np.asarray(flatten(res2))  # flatten
        assert (np.any(np.isnan(x)) == False)
        assert (np.any(np.isnan(x2)) == False)
        assert (np.any(np.isinf(x)) == False)
        assert (np.any(np.isinf(x2)) == False)
        features = np.concatenate((x, x2))

        names = ['subtraction-' + n for n in names]
        names2 = ['multiplication-' + n for n in names2]
        allNames = names + names2
        self.featureNames = ['{}:{}'.format(i, n) for i, n in enumerate(allNames)]

        return features

    # adding a comfortable function, where one can easily introduce the data
    def add_allData(self, mydata, labels):
        self.mydata = mydata
        self.labels = labels

    def train(self, withFeatureImportance=False):
        logger.info(
        "Training classifier from {} positive and {} negative labels".format(np.count_nonzero(np.asarray(self.labels)),
                                                                             len(self.labels) - np.count_nonzero(
                                                                                 np.asarray(self.labels))))
        logger.info("Training classifier from a feature vector of length {}".format(self.mydata.shape))

        if withFeatureImportance:
            oob, featImportance = self.rf.learnRFWithFeatureSelection(
                self.mydata.astype("float32"),
                (np.asarray(self.labels)).astype("uint32").reshape(-1, 1))
            logger.debug("RF feature importance: {}".format(featImportance))
            logger.debug('Feature names: {}'.format(self.featureNames))
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
    import argparse

    parser = argparse.ArgumentParser(description="trainRF",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("filepath",
                        help="read ground truth from this folder", metavar="FILE")
    parser.add_argument("rawimage_filename",
                        help="filepath+name of the raw image", metavar="FILE")
    parser.add_argument("--rawimage-h5-path", dest='rawimage_h5_path', type=str,
                        help="Path inside the rawimage HDF5 file", default='volume/data')
    parser.add_argument("initFrame", default=0, type=int,
                        help="where to begin reading the frames")
    parser.add_argument("endFrame", default=0, type=int,
                        help="where to end frames")
    parser.add_argument("outputFilename",
                        help="save RF into file", metavar="FILE")
    parser.add_argument("--filename-zero-padding", dest='filename_zero_padding', default=5, type=int,
                        help="Number of digits each file name should be long")
    parser.add_argument("--time-axis-index", dest='time_axis_index', default=2, type=int,
                        help="Zero-based index of the time axis in your raw data. E.g. if it has shape (x,t,y,c) "
                             "this value is 1. Set to -1 to disable any changes")
    parser.add_argument("--verbose", dest='verbose', action='store_true', default=False)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    filepath = args.filepath
    rawimage_filename = args.rawimage_filename
    initFrame = args.initFrame
    endFrame = args.endFrame
    fileFormatString = '{' + ':0{}'.format(args.filename_zero_padding) + '}.h5'

    rawimage = vigra.impex.readHDF5(rawimage_filename, args.rawimage_h5_path)
    try:
        print(rawimage.axistags)
    except:
        pass
    # transform such that the order is the following: X,Y,(Z),T, C
    if args.time_axis_index != -1:
        rawimage = np.rollaxis(rawimage, args.time_axis_index, -1)
    logger.info('Done loading raw data')


    # compute features
    trackingPluginManager = TrackingPluginManager()
    features = compute_features(rawimage,
                                read_in_images(initFrame, endFrame, filepath, fileFormatString),
                                initFrame,
                                endFrame,
                                trackingPluginManager)
    logger.info('Done computing features')

    selectedFeatures = find_features_without_NaNs(features)
    pos_labels = read_positiveLabels(initFrame, endFrame, filepath, fileFormatString)
    neg_labels = negativeLabels(features, pos_labels)
    numSamples = 2 * sum([len(l) for l in pos_labels]) + sum([len(l) for l in neg_labels])
    logger.info('Done extracting {} samples'.format(numSamples))
    TC = TransitionClassifier(selectedFeatures, numSamples)

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
            TC.addSample(compute_ObjFeatures(features[k], i[0]), compute_ObjFeatures(features[k + 1], i[1]), 1)
            TC.addSample(compute_ObjFeatures(features[k + 1], i[1]), compute_ObjFeatures(features[k], i[0]), 1)
        for i in neg_labels[k]:
            # negative
            logger.debug("Adding negative sample {} from pos {} to {}".format(i,
                          features[k]['RegionCenter'][i[0]], features[k + 1]['RegionCenter'][i[1]]))
            TC.addSample(compute_ObjFeatures(features[k], i[0]), compute_ObjFeatures(features[k + 1], i[1]), 0)
    logger.info('Done adding samples to RF. Beginning training...')
    TC.train()
    logger.info('Done training RF')

    srcObject = compute_ObjFeatures(features[0], 1)
    destObject = compute_ObjFeatures(features[1], 2)

    # delete file before writing
    if os.path.exists(args.outputFilename):
        os.remove(args.outputFilename)
    TC.writeRF(args.outputFilename)  # writes learned RF to disk
