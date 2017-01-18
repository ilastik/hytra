from __future__ import print_function
from __future__ import unicode_literals
import vigra
import numpy as np
import h5py
import os
import logging
from hytra.pluginsystem.plugin_manager import TrackingPluginManager
from hytra.core.ilastik_project_options import IlastikProjectOptions

def getLogger():
    ''' logger to be used in this module '''
    return logging.getLogger(__name__)

class RandomForestClassifier:
    """
    A random forest (RF) classifier wraps a list of RFs as used in ilastik,
    and allows to read the RFs trained by ilastik, as well as which features were selected.
    """

    def __init__(self, classifierPath=None, ilpFilename=None, ilpOptions=IlastikProjectOptions(), selectedFeatures=[]):
        """
        Construct a random forest by either loading it from file (`classifierPath` and `ilpFilename` must be given),
        or an empty untrained random forest with specified `selectedFeatures`
        """
        self._options = ilpOptions
        self._classifierPath = classifierPath
        self._ilpFilename = ilpFilename
        if ilpFilename is not None and classifierPath is not None:
            self._randomForests = self._readRandomForests()
            self.selectedFeatures = self._readSelectedFeatures()
        else:
            self._randomForests = []
            self.selectedFeatures = selectedFeatures

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
            getLogger().info(" Attempting to read {} classifier(s) in {} from {}".format(
                len([key for key in h5file[fullPath].keys() if 'Forest' in key]), self._ilpFilename, fullPath))

            for k in h5file[fullPath].keys():
                if 'Forest' in k:
                    getLogger().info(" Reading forest: {}".format( str('/'.join([fullPath, k])) ) )
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
                    # discard squared distances feature
                    if feature == 'ChildrenRatio_SquaredDistances':
                        continue
                        
                    # if feature == 'Coord<Principal<Kurtosis>>':
                    #     feature = 'Coord<Principal<Kurtosis> >'
                    # elif feature == 'Coord<Principal<Skewness>>':
                    #     feature = 'Coord<Principal<Skewness> >'

                    featureNameList.append(feature)
            return featureNameList

    def extractFeatureVector(self, featureDict, singleObject=False):
        """
        Extract the vector(s) of required features from the given feature dictionary,
        by concatenating the columns of the selected features into a matrix of new features, one row per object
        """
        featureVectors = None
        for f in self.selectedFeatures:
            if f not in featureDict:
                raise AssertionError("Feature '{}' not present in object features!".format(f))
            vec = featureDict[f]
            if len(vec.shape) == 1:
                if singleObject:
                    vec = np.expand_dims(vec, axis=0)
                else:
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
            getLogger().error(
                "Cannot predict from features of shape {} if {} features are expected".format(features.shape,
                      self._randomForests[0].featureCount()))
            print(features)
            raise AssertionError()

        # predict by summing the probabilities of all the given random forests (not in parallel - not optimized for speed)
        probabilities = np.zeros((features.shape[0], self._randomForests[0].labelCount()))
        for rf in self._randomForests:
            probabilities += rf.predictProbabilities(features.astype('float32'))

        return probabilities

    def train(self, featureMatrix, labels):
        """
        Train the random forest given feature matrix and labels
        """
        getLogger().info(
            "Training classifier from {} positive and {} negative labels".format(
                np.count_nonzero(np.asarray(labels)), len(labels) - np.count_nonzero(np.asarray(labels))))
        getLogger().info("Training classifier from a feature vector of length {}".format(featureMatrix.shape))

        self._randomForests = [vigra.learning.RandomForest()]
        oob = self._randomForests[0].learnRF(
            np.asarray(featureMatrix).astype("float32"),
            (np.asarray(labels)).astype("uint32").reshape(-1, 1))
        getLogger().info("RF trained with OOB Error {}".format(oob))
    
    def save(self, outputFilename=None, classifierPath='/'):
        """
        Save the random forest to a HDF5 file into a specified path inside the HDF5 file.

        Pass in `None` for both parameters to use the values specified in the constructor.

        """
        if outputFilename is None:
            outputFilename = self._ilpFilename
        assert(outputFilename is not None)

        if classifierPath is None:
            classifierPath = self._classifierPath

        if classifierPath == '/':
            fullPath = '/' + os.path.join(self._options.classifierForestsGroupName, 'Forest0000') 
        else:
            fullPath = os.path.join(classifierPath, self._options.classifierForestsGroupName, 'Forest0000')
        self._randomForests[0].writeHDF5(outputFilename, pathInFile=fullPath)

        if classifierPath == '/':
            selectedFeaturesPath = 'SelectedFeatures' 
        else:
            selectedFeaturesPath = os.path.join(classifierPath, 'SelectedFeatures')

        # write selected features
        with h5py.File(outputFilename, 'r+') as f:
            if selectedFeaturesPath in f:
                del f[selectedFeaturesPath]
            featureNamesH5 = f.create_group(selectedFeaturesPath)
            featureNamesH5 = featureNamesH5.create_group('Standard Object Features')
            for feature in self.selectedFeatures:
                featureNamesH5.create_group(feature)