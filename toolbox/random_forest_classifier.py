import sys
import os
import vigra
import numpy as np
import h5py
from pluginsystem.plugin_manager import TrackingPluginManager
import logging

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
            if self._classifierPath == '/':
                fullPath = '/' + self._options.classifierForestsGroupName
            else:
                fullPath = '/'.join([self._classifierPath, self._options.classifierForestsGroupName])
            randomForests = []
            logging.getLogger("RandomForestClassifier").info("trying to read {} classifiers in {} from {}".format(
                len(h5file[fullPath].keys()), self._ilpFilename, fullPath))

            for k in h5file[fullPath].keys():
                if 'Forest' in k:
                    print(str('/'.join([fullPath, k])))
                    rf = vigra.learning.RandomForest(str(self._ilpFilename), str('/'.join([fullPath, k])))
                    randomForests.append(rf)
            return randomForests

    def _readSelectedFeatures(self):
        """
        Read which features were selected when training this RF
        """
        with h5py.File(self._ilpFilename, 'r') as h5file:
            if self._classifierPath == '/':
                fullPath = '/' + self._options.selectedFeaturesGroupName
            else:
                fullPath = '/'.join([self._classifierPath, self._options.selectedFeaturesGroupName])
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
                if len(vec.shape) == 3:
                    for row in range(vec.shape[2]):
                        featureVectors = np.hstack([featureVectors, vec[..., row]])
                elif len(vec.shape) > 3:
                    raise ValueError("Cannot deal with features of more than two dimensions yet")
                else:
                    featureVectors = np.hstack([featureVectors, vec])

        return featureVectors

    def predictProbabilities(self, features, featureDict=None):
        """
        Given a matrix of features, where each row represents one object and each column is a specific feature,
        this method predicts the probabilities for all classes that this RF knows.

        If features=None but a featureDict is given, the selected features for this random forest are automatically extracted
        """
        assert (len(self._randomForests) > 0)

        # make sure features are good
        if features is None and featureDict is not None:
            features = self.extractFeatureVector(featureDict)
        assert (len(features.shape) == 2)
        # assert(features.shape[1] == self._randomForests[0].featureCount())
        if not features.shape[1] == self._randomForests[0].featureCount():
            logging.getLogger("RandomForestClassifier").error(
                "Cannot predict from features of shape {} if {} features are expected".format(features.shape,
                      self._randomForests[0].featureCount()))
            print(features)
            raise AssertionError()

        # predict by summing the probabilities of all the given random forests (not in parallel - not optimized for speed)
        probabilities = np.zeros((features.shape[0], self._randomForests[0].labelCount()))
        for rf in self._randomForests:
            probabilities += rf.predictProbabilities(features.astype('float32'))

        return probabilities