import numpy as np
import logging
import time
import concurrent.futures

import hytra.core.divisionfeatures
from hytra.util.progressbar import ProgressBar
from hytra.pluginsystem.plugin_manager import TrackingPluginManager
from hytra.core.random_forest_classifier import RandomForestClassifier
from hytra.core.ilastik_project_options import IlastikProjectOptions


logger = logging.getLogger("ProbabilityGenerator")


class Traxel(object):
    """
    A simple Python variant of the C++ traxel with the same interface 
    of the one of pgmlink so it can act as drop-in replacement.

    A `Traxel` is a detection with unique label `Id` at a certain `Timestep`,  with position
    (`X`,`Y`,`Z`) and `Features`
    """

    def __init__(self):
        self._scale = np.array([1, 1, 1])
        self.Id = None
        self.Timestep = None

        # dictionary of a np.array per feature (keys should be strings!)
        self.Features = {}

        # conflicting traxel ids in the same frame
        self.conflictingTraxelIds = None

    def set_x_scale(self, val):
        self._scale[0] = val

    def set_y_scale(self, val):
        self._scale[1] = val

    def set_z_scale(self, val):
        self._scale[2] = val

    def X(self):
        return self.Features['com'][0]

    def Y(self):
        return self.Features['com'][1]

    def Z(self):
        try:
            return self.Features['com'][2]
        except:
            return 0.0

    def add_feature_array(self, name, length):
        self.Features[name] = np.zeros(length)

    def set_feature_value(self, name, index, value):
        assert name in self.Features
        self.Features[name][index] = value

    def get_feature_value(self, name, index):
        assert name in self.Features
        return self.Features[name][index]

    def print_available_features(self):
        print(self.Features.keys())

    def __repr__(self):
        return "Traxel(Timestep={},Id={})".format(self.Timestep, self.Id)


def computeRegionFeaturesOnCloud(frame,
                                 rawImageFilename,
                                 rawImagePath,
                                 rawImageAxes,
                                 labelImageFilename,
                                 labelImagePath,
                                 turnOffFeatures,
                                 pluginPaths=['hytra/plugins'],
                                 featuresPerFrame = None,
                                 imageProviderPluginName='LocalImageLoader',
                                 featureSerializerPluginName='LocalFeatureSerializer'
                                ):
    '''
    Allow to use dispy to schedule feature computation to nodes running a dispynode,
    or to use multiprocessing.

    **Parameters**

    * `frame`: the frame number
    * `rawImageFilename`: the base filename of the raw image volume, or a dvid server address
    * `rawImagePath`: path inside the raw image HDF5 file, or DVID dataset UUID
    * `rawImageAxes`: axes configuration of the raw data
    * `labelImageFilename`: the base filename of the label image volume, or a dvid server address
    * `labelImagePath`: path inside the label image HDF5 file, or DVID dataset UUID
    * `pluginPaths`: where all yapsy plugins are stored (should be absolute for DVID)

    **returns** the feature dictionary for this frame if `featureSerializerPluginName == 'LocalFeatureSerializer'`
    and `featuresPerFrame == None`.
    '''

    # set up plugin manager
    from hytra.pluginsystem.plugin_manager import TrackingPluginManager
    pluginManager = TrackingPluginManager(pluginPaths=pluginPaths, turnOffFeatures=turnOffFeatures, verbose=False)
    pluginManager.setImageProvider(imageProviderPluginName)
    pluginManager.setFeatureSerializer(featureSerializerPluginName)

    # load raw and label image (depending on chosen plugin this works via DVID or locally)
    rawImage = pluginManager.getImageProvider().getImageDataAtTimeFrame(
        rawImageFilename, rawImagePath, rawImageAxes, frame)
    labelImage = pluginManager.getImageProvider().getLabelImageForFrame(
        labelImageFilename, labelImagePath, frame)

    # untwist axes, if just x and y are messed up
    if rawImage.shape[0] == labelImage.shape[1] and rawImage.shape[1] == labelImage.shape[0]:
        labelImage = np.transpose(labelImage, axes=[1, 0])

    # compute features
    moreFeats, ignoreNames = pluginManager.applyObjectFeatureComputationPlugins(
        len(labelImage.shape), rawImage, labelImage, frame, rawImageFilename)

    # combine into one dictionary
    # WARNING: if there are multiple features with the same name, they will be overwritten!
    frameFeatureItems = []
    for f in moreFeats:
        frameFeatureItems = frameFeatureItems + list(f.items())
    frameFeatures = dict(frameFeatureItems)

    # delete all ignored features
    for k in ignoreNames:
        if k in frameFeatures.keys():
            del frameFeatures[k]

    # return or save features
    if featuresPerFrame is None and featureSerializerPluginName in [u'LocalFeatureSerializer', 'LocalFeatureSerializer']:
        # simply return resulting dict
        return frame, frameFeatures
    else:
        # set up feature serializer (local or DVID for now)
        featureSerializer = pluginManager.getFeatureSerializer()

        # server address and uuid are only used by the DVID serializer
        featureSerializer.server_address = labelImageFilename
        featureSerializer.uuid = labelImagePath

        # feature dictionary used by local serializer
        featureSerializer.features_per_frame = featuresPerFrame

        # store
        featureSerializer.storeFeaturesForFrame(frameFeatures, frame)

def computeDivisionFeaturesOnCloud(frameT,
                                   featuresAtT,
                                   featuresAtTPlus1,
                                   imageProviderPlugin,
                                   labelImageFilename,
                                   labelImagePath,
                                   numDimensions,
                                   divisionFeatureNames):
    '''
    Allow to compute division features using multiprocessing

    **Parameters**

    * `frameT`: the frame number
    * `featuresAtT`: the feature dict of the current frame
    * `featuresAtTPlus1`: feature dict of next frame
    * `imageProviderPlugin`: plugin for feature loading
    * `numDimensions`: number of dimensions of the dataset
    * `divisionFeatureNames`: list of feature names for the `hytra.divisionfeatures.FeatureManager`

    **returns** a tuple of `frameT` and the dictionary of the newly computed division 
    features for `frameT`
    '''

    # get the label image of the next frame
    if frameT + 1 < imageProviderPlugin.getTimeRange(labelImageFilename, labelImagePath)[1]:
        labelImageAtTPlus1 = imageProviderPlugin.getLabelImageForFrame(labelImageFilename, labelImagePath, frameT + 1)

    # compute features
    fm = hytra.core.divisionfeatures.FeatureManager(ndim=numDimensions)
    feats = fm.computeFeatures_at(featuresAtT, featuresAtTPlus1, labelImageAtTPlus1, divisionFeatureNames, labelImageFilename)

    return frameT, feats


class DummyExecutor(object):
    """
    Class that mimics the API of concurrent.futures.ProcessPoolExecutor and 
    concurrent.futures.ThreadPoolExecutor, so that the methods can be called locally
    without threading or processing as well.
    """

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        ''' implementing enter and exit methods allows to use the `with` statement '''
        return self

    def __exit__(self, *args):
        ''' Returning false means exceptions are propagated as always '''
        return False

    def submit(self, func, *args, **kwargs):
        # create a concurrent.futures.Future to store result
        f = concurrent.futures.Future()
        f.set_running_or_notify_cancel()

        # run func
        result = func(*args, **kwargs)
        f.set_result(result)

        return f

class ProbabilityGenerator(object):
    """
    The ProbabilityGenerator contains a dictionary of all traxels. The traxels themself contain the 
    probability estimates for detection and division. Any derived class will need to populate this
    dictionary.
    """
    def __init__(self):
        self.TraxelsPerFrame = {}


class IlpProbabilityGenerator(ProbabilityGenerator):
    """
    The IlpProbabilityGenerator is a python wrapper around pgmlink's C++ traxelstore,
    but with the functionality to compute all region features 
    and evaluate the division/count/transition classifiers.
    """

    def __init__(self, 
                 ilpOptions, 
                 turnOffFeatures=[], 
                 useMultiprocessing=True, 
                 pluginPaths=['hytra/plugins'],
                 verbose=False):
        self._useMultiprocessing = useMultiprocessing
        self._options = ilpOptions
        self._pluginPaths = pluginPaths
        self._pluginManager = TrackingPluginManager(turnOffFeatures=turnOffFeatures, 
                                                    verbose=verbose,
                                                    pluginPaths=pluginPaths)
        self._pluginManager.setImageProvider(ilpOptions.imageProviderName)
        self._pluginManager.setFeatureSerializer(ilpOptions.featureSerializerName)

        self._countClassifier = None
        self._divisionClassifier = None
        self._transitionClassifier = None

        self._loadClassifiers()

        self.shape, self.timeRange = self._getShapeAndTimeRange()

        # set default division feature names
        self._divisionFeatureNames = ['ParentChildrenRatio_Count',
                                      'ParentChildrenRatio_Mean',
                                      'ChildrenRatio_Count',
                                      'ChildrenRatio_Mean',
                                      'ParentChildrenAngle_RegionCenter',
                                      'ChildrenRatio_SquaredDistances']

        # other parameters that one might want to set
        self.x_scale = 1.0
        self.y_scale = 1.0
        self.z_scale = 1.0
        self.divisionProbabilityFeatureName = 'divProb'
        self.detectionProbabilityFeatureName = 'detProb'

        self.TraxelsPerFrame = {}
        ''' this public variable contains all traxels if we're not using pgmlink '''
    
    def _loadClassifiers(self):
        if self._options.objectCountClassifierPath != None and self._options.objectCountClassifierFilename != None:
            self._countClassifier = RandomForestClassifier(self._options.objectCountClassifierPath,
                                                           self._options.objectCountClassifierFilename, self._options)
        if self._options.divisionClassifierPath != None and self._options.divisionClassifierFilename != None:
            self._divisionClassifier = RandomForestClassifier(self._options.divisionClassifierPath,
                                                              self._options.divisionClassifierFilename, self._options)
        if self._options.transitionClassifierPath != None and self._options.transitionClassifierFilename != None:
            self._transitionClassifier = RandomForestClassifier(self._options.transitionClassifierPath,
                                                                self._options.transitionClassifierFilename, self._options)
    
    def __getstate__(self):
        '''
        We define __getstate__ and __setstate__ to exclude the random forests from being pickled,
        as that is not allowed.

        See https://docs.python.org/3/library/pickle.html#pickle-state for more details.
        '''
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state['_countClassifier']
        del state['_divisionClassifier']
        del state['_transitionClassifier']
        return state

    def __setstate__(self, state):
        # Restore instance attributes
        self.__dict__.update(state)
        # Restore the random forests by reading them from scratch
        self._loadClassifiers()

    def computeRegionFeatures(self, rawImage, labelImage, frameNumber):
        """
        Computes all region features for all objects in the given image
        """
        assert (labelImage.dtype == np.uint32)

        moreFeats, ignoreNames = self._pluginManager.applyObjectFeatureComputationPlugins(len(labelImage.shape),
                                                                                          rawImage,
                                                                                          labelImage,
                                                                                          frameNumber,
                                                                                          self._options.rawImageFilename)
        frameFeatureItems = []
        for f in moreFeats:
            frameFeatureItems = frameFeatureItems + f.items()
        frameFeatures = dict(frameFeatureItems)

        # delete the "Global<Min/Max>" features as they are not nice when iterating over everything
        for k in ignoreNames:
            if k in frameFeatures.keys():
                del frameFeatures[k]

        return frameFeatures

    def computeDivisionFeatures(self, featuresAtT, featuresAtTPlus1, labelImageAtTPlus1):
        """
        Computes the division features for all objects in the images
        """
        fm = hytra.core.divisionfeatures.FeatureManager(ndim=self.getNumDimensions())
        return fm.computeFeatures_at(featuresAtT, featuresAtTPlus1, labelImageAtTPlus1, self._divisionFeatureNames)

    def setDivisionFeatures(self, divisionFeatures):
        """
        Set which features should be computed explicitly for divisions by giving a list of strings.
        Each string could be a combination of <operation>_<feature>, where Operation is one of:
            * ParentIdentity
            * SquaredDistances
            * ChildrenRatio
            * ParentChildrenAngle
            * ParentChildrenRatio

        And <feature> is any region feature plus "SquaredDistances"
        """
        # TODO: check that the strings are valid?
        self._divisionFeatureNames = divisionFeatures

    def getNumDimensions(self):
        """
        Compute the number of dimensions which is the number of axis with more than 1 element
        """
        return np.count_nonzero(np.array(self.shape) != 1)

    def _getShapeAndTimeRange(self):
        """
        extract the shape from the labelimage
        """
        shape = self._pluginManager.getImageProvider().getImageShape(
            self._options.labelImageFilename, self._options.labelImagePath)
        timerange = self._pluginManager.getImageProvider().getTimeRange(
            self._options.labelImageFilename, self._options.labelImagePath)
        return shape, timerange

    def getLabelImageForFrame(self, timeframe):
        """
        Get the label image(volume) of one time frame
        """
        rawImage = self._pluginManager.getImageProvider().getLabelImageForFrame(
            self._options.labelImageFilename, self._options.labelImagePath, timeframe)
        return rawImage

    def getRawImageForFrame(self, timeframe):
        """
        Get the raw image(volume) of one time frame
        """
        rawImage = self._pluginManager.getImageProvider().getImageDataAtTimeFrame(
            self._options.rawImageFilename, self._options.rawImagePath, timeframe)
        return rawImage

    def _extractFeaturesForFrame(self, timeframe):
        """
        extract the features of one frame, return a dictionary of features,
        where each feature vector contains N entries per object 
        (where N is the dimensionality of the feature)
        """
        rawImage = self.getRawImageForFrame(timeframe)
        labelImage = self.getLabelImageForFrame(timeframe)

        return timeframe, self.computeRegionFeatures(rawImage, labelImage, timeframe)

    def _extractDivisionFeaturesForFrame(self, timeframe, featuresPerFrame):
        """
        extract Division Features for one frame, and store them in the given featuresPerFrame dict
        """
        feats = {}
        if timeframe + 1 < self.timeRange[1]:
            labelImageAtTPlus1 = self.getLabelImageForFrame(timeframe + 1)
            feats = self.computeDivisionFeatures(featuresPerFrame[timeframe],
                                                 featuresPerFrame[timeframe + 1],
                                                 labelImageAtTPlus1)

        return timeframe, feats

    def _extractAllFeatures(self, dispyNodeIps=[], turnOffFeatures=[]):
        """
        Extract the features of all frames. 

        If a list of IP addresses is given e.g. as `dispyNodeIps = ["104.197.178.206","104.196.46.138"]`, 
        then the computation will be distributed across these nodes. Otherwise, multiprocessing will
        be used if `self._useMultiprocessing=True`, which it is by default.

        If `dispyNodeIps` is an empty list, then the feature extraction will be parallelized via
        multiprocessing.

        **TODO:** fix division feature computation for distributed mode
        """
        import logging
        # configure progress bar
        numSteps = self.timeRange[1] - self.timeRange[0]
        if self._divisionClassifier is not None:
            numSteps *= 2

        t0 = time.time()

        if(len(dispyNodeIps) == 0):
            # no dispy node IDs given, parallelize object feature computation via processes

            if self._useMultiprocessing:
                # use ProcessPoolExecutor, which instanciates as many processes as there CPU cores by default
                ExecutorType = concurrent.futures.ProcessPoolExecutor
                logging.getLogger('Traxelstore').info('Parallelizing feature extraction via multiprocessing on all cores!')
            else:
                ExecutorType = DummyExecutor
                logging.getLogger('Traxelstore').info('Running feature extraction on single core!')

            featuresPerFrame = {}
            progressBar = ProgressBar(stop=numSteps)
            progressBar.show(increase=0)

            with ExecutorType() as executor:
                # 1st pass for region features
                jobs = []
                for frame in range(self.timeRange[0], self.timeRange[1]):
                    jobs.append(executor.submit(computeRegionFeaturesOnCloud,
                                                frame,
                                                self._options.rawImageFilename, 
                                                self._options.rawImagePath,
                                                self._options.rawImageAxes,
                                                self._options.labelImageFilename,
                                                self._options.labelImagePath,
                                                turnOffFeatures,
                                                self._pluginPaths
                    ))
                for job in concurrent.futures.as_completed(jobs):
                    progressBar.show()
                    frame, feats = job.result()
                    featuresPerFrame[frame] = feats

                # 2nd pass for division features
                if self._divisionClassifier is not None:
                    jobs = []
                    for frame in range(self.timeRange[0], self.timeRange[1] - 1):
                        jobs.append(executor.submit(computeDivisionFeaturesOnCloud,
                                                    frame,
                                                    featuresPerFrame[frame],
                                                    featuresPerFrame[frame + 1],
                                                    self._pluginManager.getImageProvider(),
                                                    self._options.labelImageFilename,
                                                    self._options.labelImagePath,
                                                    self.getNumDimensions(),
                                                    self._divisionFeatureNames
                        ))

                    for job in concurrent.futures.as_completed(jobs):
                        progressBar.show()
                        frame, feats = job.result()
                        featuresPerFrame[frame].update(feats)

            # # serialize features??
            # for frame in range(self.timeRange[0], self.timeRange[1]):
            #     featureSerializer.storeFeaturesForFrame(featuresPerFrame[frame], frame)
        else:

            import logging
            logging.getLogger('Traxelstore').warning('Parallelization with dispy is WORK IN PROGRESS!')
            import random
            import dispy
            cluster = dispy.JobCluster(computeRegionFeaturesOnCloud,
                                       nodes=dispyNodeIps,
                                       loglevel=logging.DEBUG,
                                       depends=[self._pluginManager],
                                       secret="teamtracking")

            jobs = []
            for frame in range(self.timeRange[0], self.timeRange[1]):
                job = cluster.submit(frame,
                                     self._options.rawImageFilename,
                                     self._options.rawImagePath,
                                     self._options.rawImageAxes,
                                     self._options.labelImageFilename,
                                     self._options.labelImagePath,
                                     turnOffFeatures,
                                     pluginPaths=['/home/carstenhaubold/embryonic/plugins'])
                job.id = frame
                jobs.append(job)

            for job in jobs:
                job() # wait for job to finish
                print(job.exception)
                print(job.stdout)
                print(job.stderr)
                print(job.id)

            logging.getLogger('Traxelstore').warning('Using dispy we cannot compute division features yet!')
            # # 2nd pass for division features
            # if self._divisionClassifier is not None:
            #     for frame in range(self.timeRange[0], self.timeRange[1]):
            #         progressBar.show()
            #         featuresPerFrame[frame].update(self._extractDivisionFeaturesForFrame(frame, featuresPerFrame)[1])
        
        t1 = time.time()
        logger.info("Feature computation took {} secs".format(t1 - t0))
        
        return featuresPerFrame

    def _setTraxelFeatureArray(self, traxel, featureArray, name):
        ''' store the specified `featureArray` in a `traxel`'s feature dictionary under the specified key=`name` '''
        if isinstance(featureArray, np.ndarray):
            featureArray = featureArray.flatten()
        traxel.add_feature_array(name, len(featureArray))
        for i, v in enumerate(featureArray):
            traxel.set_feature_value(name, i, float(v))

    def fillTraxels(self, usePgmlink=True, ts=None, fs=None, dispyNodeIps=[], turnOffFeatures=[]):
        """
        Compute all the features and predict object count as well as division probabilities.
        Store the resulting information (and all other features) in the given pgmlink::TraxelStore,
        or create a new one if ts=None.

        usePgmlink: boolean whether pgmlink should be used and a pgmlink.TraxelStore and pgmlink.FeatureStore returned
        ts: an initial pgmlink.TraxelStore (only used if usePgmlink=True)
        fs: an initial pgmlink.FeatureStore (only used if usePgmlink=True)

        returns (ts, fs) but only if usePgmlink=True, otherwise it fills self.TraxelsPerFrame
        """
        if usePgmlink:
            import pgmlink
            if ts is None:
                ts = pgmlink.TraxelStore()
                fs = pgmlink.FeatureStore()
            else:
                assert (fs is not None)

        logger.info("Extracting features...")
        self._featuresPerFrame = self._extractAllFeatures(dispyNodeIps=dispyNodeIps, turnOffFeatures=turnOffFeatures)

        logger.info("Creating traxels...")
        progressBar = ProgressBar(stop=len(self._featuresPerFrame))
        progressBar.show(increase=0)

        for frame, features in self._featuresPerFrame.items():
            # predict random forests
            if self._countClassifier is not None:
                objectCountProbabilities = self._countClassifier.predictProbabilities(
                    features=None, featureDict=features)

            if self._divisionClassifier is not None and frame + 1 < self.timeRange[1]:
                divisionProbabilities = self._divisionClassifier.predictProbabilities(
                    features=None, featureDict=features)

            # create traxels for all objects
            for objectId in range(1, list(features.values())[0].shape[0]):
                # print("Frame {} Object {}".format(frame, objectId))
                pixelSize = features['Count'][objectId]
                if pixelSize == 0 or (self._options.sizeFilter is not None \
                        and (pixelSize < self._options.sizeFilter[0] \
                                     or pixelSize > self._options.sizeFilter[1])):
                    continue

                # create traxel
                if usePgmlink:
                    traxel = pgmlink.Traxel()
                else:
                    traxel = Traxel()
                traxel.Id = objectId
                traxel.Timestep = frame

                # add raw features
                for key, val in features.items():
                    if key == 'id':
                        traxel.idInSegmentation = val[objectId]
                    elif key == 'filename':
                        traxel.segmentationFilename = val[objectId]
                    else:
                        try:
                            if isinstance(val, list):  # polygon feature returns a list!
                                featureValues = val[objectId]
                            else:
                                featureValues = val[objectId, ...]
                        except:
                            logger.error(
                                "Could not get feature values of {} for key {} from matrix with shape {}".format(
                                    objectId, key, val.shape))
                            raise AssertionError()
                        try:
                            self._setTraxelFeatureArray(traxel, featureValues, key)
                            if key == 'RegionCenter':
                                self._setTraxelFeatureArray(traxel, featureValues, 'com')
                        except:
                            logger.error(
                                "Could not add feature array {} for {}".format(
                                    featureValues, key))
                            raise AssertionError()

                # add random forest predictions
                if self._countClassifier is not None:
                    self._setTraxelFeatureArray(
                        traxel, objectCountProbabilities[objectId, :], self.detectionProbabilityFeatureName)

                if self._divisionClassifier is not None and frame + 1 < self.timeRange[1]:
                    self._setTraxelFeatureArray(
                        traxel, divisionProbabilities[objectId, :], self.divisionProbabilityFeatureName)

                # set other parameters
                traxel.set_x_scale(self.x_scale)
                traxel.set_y_scale(self.y_scale)
                traxel.set_z_scale(self.z_scale)

                if usePgmlink:
                    # add to pgmlink's traxelstore
                    ts.add(fs, traxel)
                else:
                    self.TraxelsPerFrame.setdefault(frame, {})[objectId] = traxel
            progressBar.show()

        if usePgmlink:
            return ts, fs

    def getTraxelFeatureDict(self, frame, objectId):
        """
        Getter method for features per traxel
        """
        assert self._featuresPerFrame != None
        traxelFeatureDict = {}
        for k, v in self._featuresPerFrame[frame].items():
            if 'Polygon' in k:
                traxelFeatureDict[k] = v[objectId]
            else:
                traxelFeatureDict[k] = v[objectId, ...]
        return traxelFeatureDict

    def getTransitionFeatureVector(self, featureDictObjectA, featureDictObjectB, selectedFeatures):
        """
        Return component wise difference and product of the selected features as input for the TransitionClassifier
        """
        features = np.array(self._pluginManager.applyTransitionFeatureVectorConstructionPlugins(
            featureDictObjectA, featureDictObjectB, selectedFeatures))
        features = np.expand_dims(features, axis=0)
        return features


if __name__ == '__main__':
    """
    Builds a traxelstore from a given ilastik project file and the raw data as HDF5 volume
    """
    import argparse

    parser = argparse.ArgumentParser(description='Build a traxelstore from a given ilastik project',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ilastik-project', required=True, type=str, dest='ilpFilename',
                        help='Filename of the ilastik project')
    parser.add_argument('--raw', required=True, type=str, dest='rawFilename',
                        help='Filename of the hdf5 file containing the raw data')
    parser.add_argument('--raw-path', required=True, type=str, dest='rawPath',
                        help='Path inside HDF5 file to raw volume')
    parser.add_argument("--raw-data-axes", dest='rawAxes', type=str, default='txyzc',
                        help="axes ordering of the raw image, e.g. xyztc.")
    parser.add_argument('--label-image-path', type=str, dest='labelImagePath',
                        help='Path inside ilastik project file to the label image',
                        default='/TrackingFeatureExtraction/LabelImage/0000/[[%d, 0, 0, 0, 0], [%d, %d, %d, %d, 1]]')
    parser.add_argument('--label-image', type=str, dest='labelImageFilename',
                        help='Path to Label image. If empty ilp filename will be used',
                        default='')
    parser.add_argument('--object-count-classifier-path', type=str, dest='objectCountClassifierPath',
                        help='Path inside ilastik project file to the object count classifier',
                        default='/CountClassification')
    parser.add_argument('--division-classifier-path', type=str, dest='divisionClassifierPath',
                        help='Path inside ilastik project file to the division classifier',
                        default='/DivisionDetection')
    parser.add_argument('--without-divisions', dest='withoutDivisions', action='store_true',
                        help='Specify this if no divisions are allowed in this dataset',
                        default=False)
    parser.add_argument('--rf-zero-padding', type=int, dest='rfZeroPadding', default=4,
                        help='Number of digits per forest index inside the ClassifierForests HDF5 group')

    parser.add_argument('--image-provider', type=str, dest='image_provider_name', default="LocalImageLoader")
    parser.add_argument('--feature-serializer', type=str, dest='feature_serializer_name', 
                        default='LocalFeatureSerializer')
    parser.add_argument('--disable-multiprocessing', dest='disableMultiprocessing', action='store_true',
                        help='Do not use multiprocessing to speed up computation',
                        default=False)

    args = parser.parse_args()

    ilpOptions = IlastikProjectOptions()

    logging.basicConfig(level=logging.INFO)

    ilpOptions.objectCountClassifierPath = args.objectCountClassifierPath
    if args.withoutDivisions:
        ilpOptions.divisionClassifierPath = None
    else:
        ilpOptions.divisionClassifierPath = args.divisionClassifierPath
    ilpOptions.randomForestZeroPaddingWidth = args.rfZeroPadding
    ilpOptions.labelImagePath = args.labelImagePath
    ilpOptions.rawImagePath = args.rawPath
    ilpOptions.rawImageAxes = args.rawAxes

    ilpOptions.imageProviderName = args.image_provider_name
    ilpOptions.featureSerializerName = args.feature_serializer_name

    if(not args.labelImageFilename):
        ilpOptions.labelImageFilename = args.ilpFilename
    else:
        ilpOptions.labelImageFilename = args.labelImageFilename

    ilpOptions.objectCountClassifierFilename = args.ilpFilename
    ilpOptions.divisionClassifierFilename = args.ilpFilename
    ilpOptions.rawImageFilename = args.rawFilename

    probabilityGenerator = IlpProbabilityGenerator(ilpOptions=ilpOptions, useMultiprocessing=not args.disableMultiprocessing)
    probabilityGenerator.timeRange = (0, 3)
    probabilityGenerator.fillTraxels(usePgmlink=False)
