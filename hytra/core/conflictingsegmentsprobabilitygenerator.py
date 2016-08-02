import numpy as np
import logging
import time
import concurrent.futures

from hytra.core.probabilitygenerator import IlpProbabilityGenerator, computeDivisionFeaturesOnCloud, computeRegionFeaturesOnCloud, DummyExecutor
from hytra.util.progressbar import ProgressBar

def getLogger():
    return logging.getLogger(__name__)

class ConflictingSegmentsProbabilityGenerator(IlpProbabilityGenerator):
    """
    Specialization of the probability generator that computes all the features on its own,
    to have more than one segmentation hypotheses per timeframe.

    First step: make sure that objects from different hypotheses have different IDs

    * do that by adding the maxId of the "previous" segmentation hypothesis for that frame
    * store reference which hypothesis this segment comes from in Traxel, so that we can reconstruct a result from the graph and images

    """

    def __init__(self, 
                 ilpOptions,
                 additionalLabelImageFilenames,
                 additionalLabelImagePaths,
                 turnOffFeatures=[], 
                 useMultiprocessing=True, 
                 pluginPaths=['hytra/plugins'],
                 verbose=False):
        """
        """
        super(ConflictingSegmentsProbabilityGenerator, self).__init__(ilpOptions,
                                                                      turnOffFeatures,
                                                                      useMultiprocessing,
                                                                      pluginPaths,
                                                                      verbose)
                                                                      
        # store the additional segmentation hypotheses and check that they are of the same size
        self._labelImageFilenames = additionalLabelImageFilenames
        self._labelImagePaths = additionalLabelImagePaths

        for filename, path in zip(self._labelImageFilenames, self._labelImagePaths):
            assert(self._pluginManager.getImageProvider().getImageShape(filename, path) == self.shape)
            assert(self._pluginManager.getImageProvider().getTimeRange(filename, path) == self.timeRange)
        
        self._labelImageFilenames.insert(0, ilpOptions.labelImageFilename)
        self._labelImagePaths.insert(0, ilpOptions.labelImagePath)
        self._labelImageFrameIdToGlobalId = {} # map from (labelImageFilename, frame, id) to (id)
        
    def fillTraxels(self, usePgmlink=True, ts=None, fs=None, dispyNodeIps=[], turnOffFeatures=[]):
        """
        Compute all the features and predict object count as well as division probabilities.
        Store the resulting information (and all other features) inside `self.TraxelsPerFrame`.

        It also computes which of the segmentation hypotheses overlap and are mutually exclusive, and stores 
        that per traxel, in each traxel's `conflictingTraxelIds` list. (Can only conflict within the timeframe)

        WARNING: usePgmlink is not supported for this derived class, so must be `False`!
        WARNING: distributed computation via Dispy is not supported here, so dispyNodeIps must be an empty list!
        """

        assert(not usePgmlink)
        assert(len(dispyNodeIps) == 0)

        super(ConflictingSegmentsProbabilityGenerator, self).fillTraxels(usePgmlink, ts, fs, dispyNodeIps, turnOffFeatures)

        # find exclusion constraints
        for frame in range(self.timeRange[0], self.timeRange[1]):
            for labelImageIndexA in range(len(self._labelImageFilenames)):
                labelImageA = self._pluginManager.getImageProvider().getLabelImageForFrame(self._labelImageFilenames[labelImageIndexA],
                                                                                            self._labelImagePaths[labelImageIndexA],
                                                                                            frame)
                for labelImageIndexB in range(labelImageIndexA + 1, len(self._labelImageFilenames)):
                    labelImageB = self._pluginManager.getImageProvider().getLabelImageForFrame(self._labelImageFilenames[labelImageIndexB],
                                                                                               self._labelImagePaths[labelImageIndexB],
                                                                                               frame)
                    # check for overlaps - even a 1-pixel overlap is enough to be mutually exclusive!
                    for objectIdA in np.unique(labelImageA):
                        if objectIdA == 0:
                            continue
                        overlapping = set(np.unique(labelImageB[labelImageA == objectIdA])) - set([0])
                        overlappingGlobalIds = [self._labelImageFrameIdToGlobalId[(self._labelImageFilenames[labelImageIndexB], frame, o)] for o in overlapping]
                        globalIdA = self._labelImageFrameIdToGlobalId[(self._labelImageFilenames[labelImageIndexA], frame, objectIdA)]
                        if self.TraxelsPerFrame[frame][globalIdA].conflictingTraxelIds is None:
                            self.TraxelsPerFrame[frame][globalIdA].conflictingTraxelIds = []
                        self.TraxelsPerFrame[frame][globalIdA].conflictingTraxelIds.extend(overlappingGlobalIds)
                        for globalIdB in overlappingGlobalIds:
                            if self.TraxelsPerFrame[frame][globalIdB].conflictingTraxelIds is None:
                                self.TraxelsPerFrame[frame][globalIdB].conflictingTraxelIds = []
                            self.TraxelsPerFrame[frame][globalIdB].conflictingTraxelIds.append(globalIdA)
        # FIXME: right now an object that overlaps with two objects in an alternative segmentation
        #        adds a constraint that the triplet cannot be active at once, but actually the two others could be active at once!

    def _insertFilenameAndIdToFeatures(self, featureDict, filename):
        """
        For later disambiguation, we store for each row in the feature matrix which file it came from.
        We also store the label image id.  
        """

        # get num elements:
        numElements = None

        for _, v in featureDict.iteritems():
            try:
                currentNumElements = v.shape[0]
            except:
                currentNumElements = len(v)

            if numElements is None:
                numElements = currentNumElements
            else:
                assert(numElements == currentNumElements)

        featureDict['filename'] = [filename] * numElements
        featureDict['id'] = range(numElements)

    def _mergeFrameFeatures(self, originalDict, nextDict):
        """
        Merge the feature vectors of every feature (key=featureName, value=list of feature values for each object)
        The `originalDict` is altered to also contain the `nextDict`.

        Ignores the 0th element in each feature vector of nextDict
        """
        for k, v in originalDict.iteritems():
            assert(k in nextDict) # all frames should have the same features
            if isinstance(v, np.ndarray):
                originalDict[k] = np.concatenate((v, nextDict[k][1:]))
            else:
                originalDict[k].extend(nextDict[k][1:])

    def _storeBackwardMapping(self, featuresPerFrame):
        """
        populates the `self._labelImageFrameIdToGlobalId` dictionary
        """
        for frame, featureDict in featuresPerFrame.iteritems():
            for newId, (filename, objectId) in enumerate(zip(featureDict['filename'], featureDict['id'])):
                self._labelImageFrameIdToGlobalId[(filename, frame, objectId)] = newId

    def _extractAllFeatures(self, dispyNodeIps=[], turnOffFeatures=[]):
        """
        Extract the features of all frames of all segmentation hypotheses. 
        Feature extraction will be parallelized via multiprocessing.

        WARNING: distributed computation via Dispy is not supported here, so dispyNodeIps must be an empty list!
        """

        # configure progress bar
        numSteps = self.timeRange[1] - self.timeRange[0]
        if self._divisionClassifier is not None:
            numSteps *= 2

        t0 = time.time()

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
            # 1st pass for region features, once per segmentation hypotheses
            for filename, path in zip(self._labelImageFilenames, self._labelImagePaths):
                jobs = []
                for frame in range(self.timeRange[0], self.timeRange[1]):
                    jobs.append(executor.submit(computeRegionFeaturesOnCloud,
                                                frame,
                                                self._options.rawImageFilename, 
                                                self._options.rawImagePath,
                                                self._options.rawImageAxes,
                                                filename,
                                                path,
                                                turnOffFeatures,
                                                self._pluginPaths
                    ))
                for job in concurrent.futures.as_completed(jobs):
                    progressBar.show()
                    frame, feats = job.result()
                    self._insertFilenameAndIdToFeatures(feats, filename)
                    if frame not in featuresPerFrame:
                        featuresPerFrame[frame] = feats
                    else:
                        self._mergeFrameFeatures(featuresPerFrame[frame], feats)

            # 2nd pass for division features
            # TODO: the division feature manager should see the child candidates in all segmentation hypotheses
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

        self._storeBackwardMapping(featuresPerFrame)

        t1 = time.time()
        getLogger().info("Feature computation took {} secs".format(t1 - t0))
        
        return featuresPerFrame