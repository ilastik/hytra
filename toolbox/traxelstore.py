import sys
sys.path.append('../.')
sys.path.append('.')

import os
from empryonic import io
import json
from progressbar import ProgressBar
import vigra
import divisionfeatures
import numpy as np
import h5py

class IlastikProjectOptions:
    """
    The Ilastik Project Options configure where in the project HDF5 file the important things can be found.
    Use this when creating a Traxelstore
    """
    def __init__(self):
        self.objectCountClassifierFile = None
        self.objectCountClassifierPath = '/CountClassification'
        self.divisionClassifierFile = None
        self.divisionClassifierPath = '/DivisionDetection'
        self.transitionClassifierFile = None
        self.transitionClassifierPath = None
        self.selectedFeaturesGroupName = 'SelectedFeatures'
        self.classifierForestsGroupName = 'ClassifierForests'
        self.randomForestZeroPaddingWidth = 4
        self.labelImageFilename = None
        self.labelImagePath = '/TrackingFeatureExtraction/LabelImage/0000/[[%d, 0, 0, 0, 0], [%d, %d, %d, %d, 1]]'
        self.rawImageFilename = None
        self.rawImagePath = None

class RandomForestClassifier:
    """
    A random forest (RF) classifier wraps a list of RFs as used in ilastik,
    and allows to read the RFs trained by ilastik, as well as which features were selected.
    """
    def __init__(self, classifierPath, ilpFilename, ilpOptions=IlastikProjectOptions()):
        self._options = ilpOptions
        self._classifierPath = classifierPath
        self._ilpFilename = ilpFilename
        self._randomForests = self._readRandomForests()
        self.selectedFeatures = self._readSelectedFeatures()

    def _readRandomForests(self):
        """
        Read in a list of random forests at a given location in the hdf5 file
        """
        with h5py.File(self._ilpFilename, 'r') as h5file:
            fullPath = os.path.join(self._classifierPath, self._options.classifierForestsGroupName)
            randomForests = []
            print("trying to read {} classifiers in {} from {}".format(len(h5file[fullPath].keys()),self._ilpFilename, fullPath))
            for k in h5file[fullPath].keys():
                if 'Forest' in k:
                    print(str('/'.join([fullPath, k])))
                    rf = vigra.learning.RandomForest(str(self._ilpFilename), str(os.path.join(fullPath, k)))
                    randomForests.append(rf)
            return randomForests

    def _readSelectedFeatures(self):
        """
        Read which features were selected when training this RF
        """
        with h5py.File(self._ilpFilename, 'r') as h5file:
            fullPath = os.path.join(self._classifierPath, self._options.selectedFeaturesGroupName)
            featureNameList = []

            for feature_group_name in h5file[fullPath].keys():
                feature_group = h5file[fullPath][feature_group_name]
                for feature in feature_group.keys():
                    # # discard squared distances feature
                    # if feature == 'ChildrenRatio_SquaredDistances':
                    #     continue

                    # if feature == 'Coord<Principal<Kurtosis>>':
                    #     feature = 'Coord<Principal<Kurtosis> >'
                    # elif feature == 'Coord<Principal<Skewness>>':
                    #     feature = 'Coord<Principal<Skewness> >'

                    featureNameList.append(feature)
            return featureNameList

    def extractFeatureVector(self, featureDict):
        """
        Extract the vector(s) of required features from the given feature dictionary,
        by concatenating the columns of the selected features into a matrix of new features, one row per object
        """
        featureVectors = None
        for f in self.selectedFeatures:
            assert f in featureDict
            vec = featureDict[f]
            if len(vec.shape) == 1:
                vec = np.expand_dims(vec, axis=1)
            if featureVectors is None:
                featureVectors = vec
            else:
                featureVectors = np.hstack([featureVectors, vec])

        return featureVectors

    def predictProbabilities(self, features, featureDict=None):
        """
        Given a matrix of features, where each row represents one object and each column is a specific feature,
        this method predicts the probabilities for all classes that this RF knows.

        If features=None but a featureDict is given, the selected features for this random forest are automatically extracted
        """
        assert(len(self._randomForests) > 0)

        # make sure features are good
        if features is None and featureDict is not None:
            features = self.extractFeatureVector(featureDict)
        assert(len(features.shape) == 2)
        # assert(features.shape[1] == self._randomForests[0].featureCount())
        if not features.shape[1] == self._randomForests[0].featureCount():
            print("Cannot predict from features of shape {} if {} features are expected".format(features.shape, self._randomForests[0].featureCount()))
            print(features)
            raise AssertionError()

        # predict by summing the probabilities of all the given random forests (not in parallel - not optimized for speed)
        probabilities = np.zeros((features.shape[0], self._randomForests[0].labelCount()))
        for rf in self._randomForests:
            probabilities += rf.predictProbabilities(features.astype('float32'))

        return probabilities

class Traxel:
    """
    A simple Python variant of the C++ traxel with the same interface of the one of pgmlink so it can act as drop-in replacement.
    """
    def __init__(self):
        self._scale = np.array([1,1,1])
        self.Id = None
        self.Timestep = None

        # dictionary of a np.array per feature (keys should be strings!)
        self.Features = {}

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
        print self.Features.keys()

class Traxelstore:
    """
    The traxelstore is a python wrapper around pgmlink's C++ traxelstore, 
    but with the functionality to compute all region features and evaluate the division/count/transition classifiers.
    """
    def __init__(self, ilpOptions):
        assert(os.path.exists(ilpOptions.labelImageFilename))
        assert(os.path.exists(ilpOptions.rawImageFilename))
        
        self._options = ilpOptions
        self._countClassifier = None
        self._divisionClassifier = None
        self._transitionClassifier = None
        
        if ilpOptions.objectCountClassifierPath != None and ilpOptions.objectCountClassifierFilename != None:
            self._countClassifier = RandomForestClassifier(ilpOptions.objectCountClassifierPath, ilpOptions.objectCountClassifierFilename, ilpOptions)
        if ilpOptions.divisionClassifierPath != None and ilpOptions.divisionClassifierFilename != None:
            self._divisionClassifier = RandomForestClassifier(ilpOptions.divisionClassifierPath, ilpOptions.divisionClassifierFilename, ilpOptions)
        if ilpOptions.transitionClassifierPath != None and ilpOptions.transitionClassifierFilename != None:
            self._transitionClassifier = RandomForestClassifier(ilpOptions.transitionClassifierPath, ilpOptions.transitionClassifierFilename, ilpOptions)

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

        # this public variable contains all traxels if we're not using pgmlink
        self.TraxelsPerFrame = {}

    @staticmethod
    def computeRegionFeatures(rawImage, labelImage):
        """
        Computes all region features for all objects in the given image
        """
        assert(labelImage.dtype == np.uint32)
        regionFeatures = vigra.analysis.extractRegionFeatures(rawImage.squeeze().astype('float32'), labelImage.squeeze(), ignoreLabel=0)
        regionFeatures = dict(regionFeatures.items()) # transform to dict

        # delete the "Global<Min/Max>" features as they are not nice when iterating over everything
        globalKeysToDelete = [k for k in regionFeatures.keys() if 'Global' in k]
        for k in globalKeysToDelete:
            del regionFeatures[k]

        # additional features can only be computed for 2D data at the moment, so return pure region features only
        if len(labelImage.shape) > 2:
            return regionFeatures
        
        convexHullFeatures = vigra.analysis.extractConvexHullFeatures(labelImage.squeeze(), ignoreLabel=0)
        skeletonFeatures = vigra.analysis.extractSkeletonFeatures(labelImage.squeeze()) # , ignoreLabel=0 <- imposed automatically
        return dict(regionFeatures.items() + convexHullFeatures.items() + skeletonFeatures.items())

    def computeDivisionFeatures(self, featuresAtT, featuresAtTPlus1, labelImageAtTPlus1):
        """
        Computes the division features for all objects in the images
        """
        fm = divisionfeatures.FeatureManager(ndim=self.getNumDimensions())
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
        with h5py.File(self._options.labelImageFilename, 'r') as h5file:
            shape = h5file['/'.join(self._options.labelImagePath.split('/')[:-1])].values()[0].shape[1:4]
            maxTime = len(h5file['/'.join(self._options.labelImagePath.split('/')[:-1])].keys())
            return shape, (0, maxTime)

    def getLabelImageForFrame(self, timeframe):
        """
        Get the label image(volume) of one time frame
        """
        with h5py.File(self._options.labelImageFilename, 'r') as h5file:
            labelImage = h5file[self._options.labelImagePath % (timeframe, timeframe+1, self.shape[0], self.shape[1], self.shape[2])][0, ..., 0].squeeze().astype(np.uint32)
            return labelImage

    def getRawImageForFrame(self, timeframe):
        """
        Get the raw image(volume) of one time frame
        """
        with h5py.File(self._options.rawImageFilename, 'r') as rawH5:
            rawImage = rawH5[self._options.rawImagePath][timeframe, ...]
            return rawImage

    def _extractFeaturesForFrame(self, timeframe):
        """
        extract the features of one frame, return a dictionary of features, 
        where each feature vector contains N entries per object (where N is the dimensionality of the feature)
        """
        rawImage = self.getRawImageForFrame(timeframe)
        labelImage = self.getLabelImageForFrame(timeframe)

        return Traxelstore.computeRegionFeatures(rawImage, labelImage)

    def _extractDivisionFeaturesForFrame(self, timeframe, featuresPerFrame):
        """
        extract Division Features for one frame, and store them in the given featuresPerFrame dict
        """
        if timeframe + 1 < self.timeRange[1]:
            labelImageAtTPlus1 = self.getLabelImageForFrame(timeframe + 1)
            featuresPerFrame[timeframe].update(self.computeDivisionFeatures(featuresPerFrame[timeframe], featuresPerFrame[timeframe + 1], labelImageAtTPlus1))

    def _extractAllFeatures(self):
        """
        extract the features of all frames
        """
        # configure progress bar
        numSteps = self.timeRange[1] - self.timeRange[0]
        if self._divisionClassifier is not None:
            numSteps *= 2
        progressBar = ProgressBar(stop=numSteps)
        progressBar.show(increase=0)

        # 1st pass for region features
        featuresPerFrame = {}
        for frame in range(self.timeRange[0], self.timeRange[1]):
            progressBar.show()
            featuresPerFrame[frame] = self._extractFeaturesForFrame(frame)

        # 2nd pass for division features
        if self._divisionClassifier is not None:
            for frame in range(self.timeRange[0], self.timeRange[1]):
                progressBar.show()
                self._extractDivisionFeaturesForFrame(frame, featuresPerFrame)

        return featuresPerFrame

    def _setTraxelFeatureArray(self, traxel, featureArray, name):
        featureArray = featureArray.flatten()
        traxel.add_feature_array(name, len(featureArray))
        for i, v in enumerate(featureArray):
            traxel.set_feature_value(name, i, float(v))

    def fillTraxelStore(self, usePgmlink=True, ts=None, fs=None):
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
                assert(fs is not None)

        print("Extracting features...")
        self._featuresPerFrame = self._extractAllFeatures()

        print("Creating traxels...")
        progressBar = ProgressBar(stop=len(self._featuresPerFrame))
        progressBar.show(increase=0)

        for frame, features in self._featuresPerFrame.iteritems():
            for objectId in range(1, features.values()[0].shape[0]):
                # print("Frame {} Object {}".format(frame, objectId))

                # create traxel
                if usePgmlink:
                    traxel = pgmlink.Traxel()
                else:
                    traxel = Traxel()
                traxel.Id = objectId
                traxel.Timestep = frame

                # add raw features
                for key, val in features.iteritems():
                    try:
                        if isinstance(val, list): # polygon feature returns a list!
                            featureValues = val[objectId]
                        else:
                            featureValues = val[objectId,...]
                    except:
                        print("Could not get feature values of {} for key {} from matrix with shape {}".format(objectId, key, val.shape))
                        sys.exit()
                    try:
                        self._setTraxelFeatureArray(traxel, featureValues, key)
                        if key == 'RegionCenter':
                            self._setTraxelFeatureArray(traxel, featureValues, 'com')
                    except:
                        print("Could not add feature array {} of shape {} for {}".format(featureValues, featureValues.shape, key))
                        sys.exit()

                # add random forest predictions
                if self._countClassifier is not None:
                    probs = self._countClassifier.predictProbabilities(features=None, featureDict=features)
                    self._setTraxelFeatureArray(traxel, probs, self.detectionProbabilityFeatureName)

                if self._divisionClassifier is not None:
                    probs = self._divisionClassifier.predictProbabilities(features=None, featureDict=features)
                    self._setTraxelFeatureArray(traxel, probs, self.divisionProbabilityFeatureName)

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

    def getTransitionProbability(self, timeframeA, objectIdA, timeframeB, objectIdB):
        """
        Evaluate the transition classifier for the two given objects, 
        as this probability doesn't go into pgmlink's traxelstore.
        """
        raise NotImplementedError()

    def getTraxelFeatureDict(self, frame, objectId):
        """
        Getter method for features per traxel
        """
        assert self._featuresPerFrame != None
        traxelFeatureDict = {}
        for k, v in self._featuresPerFrame[frame].iteritems():
            if 'Polygon' in k:
                traxelFeatureDict[k] = v[objectId]
            else:
                traxelFeatureDict[k] = v[objectId, ...]
        return traxelFeatureDict

    def getTransitionFeatureVector(self, featureDictObjectA, featureDictObjectB, selectedFeatures):
        """
        Return component wise difference and product of the selected features as input for the TransitionClassifier
        """
        from compiler.ast import flatten
        res=[]
        res2=[]

        for key in selectedFeatures:
            if key == "Global<Maximum >" or key=="Global<Minimum >":
                # the global min/max intensity is not interesting
                continue
            elif key == 'RegionCenter':
                res.append(np.linalg.norm(featureDictObjectA[key]-featureDictObjectB[key])) #difference of features
                res2.append(np.linalg.norm(featureDictObjectA[key]*featureDictObjectB[key])) #product of features
            elif key == 'Histogram': #contains only zeros, so trying to see what the prediction is without it
                continue
            elif key == 'Polygon': #vect has always another length for different objects, so center would be relevant
                continue
            else:
                res.append((featureDictObjectA[key]-featureDictObjectB[key]).tolist() )  #prepare for flattening
                res2.append((featureDictObjectA[key]*featureDictObjectB[key]).tolist() )  #prepare for flattening
            
        x= np.asarray(flatten(res)) #flatten
        x2= np.asarray(flatten(res2)) #flatten
        assert(np.any(np.isnan(x)) == False)
        assert(np.any(np.isnan(x2)) == False)
        assert(np.any(np.isinf(x)) == False)
        assert(np.any(np.isinf(x2)) == False)
        
        features = np.concatenate((x,x2))
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
    parser.add_argument('--label-image-path', type=str, dest='labelImagePath',
                        help='Path inside ilastik project file to the label image',
                        default='/TrackingFeatureExtraction/LabelImage/0000/[[%d, 0, 0, 0, 0], [%d, %d, %d, %d, 1]]')
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

    args = parser.parse_args()

    ilpOptions = IlastikProjectOptions()

    ilpOptions.objectCountClassifierPath = args.objectCountClassifierPath
    if args.withoutDivisions:
        ilpOptions.divisionClassifierPath = None
    else:
        ilpOptions.divisionClassifierPath = args.divisionClassifierPath
    ilpOptions.randomForestZeroPaddingWidth = args.rfZeroPadding
    ilpOptions.labelImagePath = args.labelImagePath
    ilpOptions.rawImagePath = args.rawPath
    ilpOptions.labelImageFilename = args.ilpFilename
    ilpOptions.objectCountClassifierFilename = args.ilpFilename
    ilpOptions.divisionClassifierFilename = args.ilpFilename
    ilpOptions.rawImageFilename = args.rawFilename

    traxelstore = Traxelstore(ilpOptions=ilpOptions)
    traxelstore.timeRange = (0,2)
    ts, fs = traxelstore.fillTraxelStore()

