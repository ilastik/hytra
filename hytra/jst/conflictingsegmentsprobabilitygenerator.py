from __future__ import print_function, absolute_import, nested_scopes, generators, division, with_statement, unicode_literals
import numpy as np
import logging
import time
import concurrent.futures

from hytra.core.probabilitygenerator import IlpProbabilityGenerator, computeDivisionFeaturesOnCloud, computeRegionFeaturesOnCloud, DummyExecutor
from hytra.util.progressbar import ProgressBar

def getLogger():
    return logging.getLogger(__name__)

def findConflictingHypothesesInSeparateProcess(frame,
                                               labelImageFilenames,
                                               labelImagePaths,
                                               labelImageFrameIdToGlobalId,
                                               pluginPaths=['hytra/plugins'],
                                               imageProviderPluginName='LocalImageLoader'):
    """
    Look which objects between different segmentation hypotheses (given as different labelImages)
    overlap, and return a dictionary of those overlapping situations.

    Meant to be run in its own process using `concurrent.futures.ProcessPoolExecutor`
    """

    # set up plugin manager
    from hytra.pluginsystem.plugin_manager import TrackingPluginManager
    pluginManager = TrackingPluginManager(pluginPaths=pluginPaths, verbose=False)
    pluginManager.setImageProvider(imageProviderPluginName)

    overlaps = {} # overlap dict: key=globalId, value=[list of globalIds]

    for labelImageIndexA in range(len(labelImageFilenames)):
        labelImageA = pluginManager.getImageProvider().getLabelImageForFrame(labelImageFilenames[labelImageIndexA],
                                                                                    labelImagePaths[labelImageIndexA],
                                                                                    frame)
        for labelImageIndexB in range(labelImageIndexA + 1, len(labelImageFilenames)):
            labelImageB = pluginManager.getImageProvider().getLabelImageForFrame(labelImageFilenames[labelImageIndexB],
                                                                                        labelImagePaths[labelImageIndexB],
                                                                                        frame)
            # check for overlaps - even a 1-pixel overlap is enough to be mutually exclusive!
            for objectIdA in np.unique(labelImageA):
                if objectIdA == 0:
                    continue
                overlapping = set(np.unique(labelImageB[labelImageA == objectIdA])) - set([0])
                overlappingGlobalIds = [labelImageFrameIdToGlobalId[(labelImageFilenames[labelImageIndexB], frame, o)] for o in overlapping]
                globalIdA = labelImageFrameIdToGlobalId[(labelImageFilenames[labelImageIndexA], frame, objectIdA)]
                overlaps.setdefault(globalIdA, []).extend(overlappingGlobalIds)
                for globalIdB in overlappingGlobalIds:
                    overlaps.setdefault(globalIdB, []).append(globalIdA)

    return frame, overlaps

def computeJaccardScoresOnCloud(frame,
                                labelImageFilenames,
                                labelImagePaths,
                                labelImageFrameIdToGlobalId,
                                groundTruthFilename,
                                groundTruthPath,
                                groundTruthMinJaccardScore,
                                pluginPaths=['hytra/plugins'],
                                imageProviderPluginName='LocalImageLoader'):
    """
    Compute jaccard scores of all objects in the different segmentations with the ground truth for that frame.
    Returns a dictionary of overlapping GT labels and the score per globalId in that frame, as well as 
    a dictionary specifying the matching globalId and score for every GT label (as a list ordered by score, best match last).

    Meant to be run in its own process using `concurrent.futures.ProcessPoolExecutor`
    """

    # set up plugin manager
    from hytra.pluginsystem.plugin_manager import TrackingPluginManager
    pluginManager = TrackingPluginManager(pluginPaths=pluginPaths, verbose=False)
    pluginManager.setImageProvider(imageProviderPluginName)

    scores = {}
    gtToGlobalIdMap = {}

    groundTruthLabelImage = pluginManager.getImageProvider().getLabelImageForFrame(groundTruthFilename, groundTruthPath, frame)

    for labelImageIndexA in range(len(labelImageFilenames)):
        labelImageA = pluginManager.getImageProvider().getLabelImageForFrame(labelImageFilenames[labelImageIndexA],
                                                                                    labelImagePaths[labelImageIndexA],
                                                                                    frame)
        # check for overlaps - even a 1-pixel overlap is enough to be mutually exclusive!
        for objectIdA in np.unique(labelImageA):
            if objectIdA == 0:
                continue
            globalIdA = labelImageFrameIdToGlobalId[(labelImageFilenames[labelImageIndexA], frame, objectIdA)]
            overlap = groundTruthLabelImage[labelImageA == objectIdA]
            overlappingGtElements = set(np.unique(overlap)) - set([0])
            
            for gtLabel in overlappingGtElements:
                # compute Jaccard scores
                intersectingPixels = np.sum(overlap == gtLabel)
                unionPixels = np.sum(np.logical_or(groundTruthLabelImage == gtLabel, labelImageA == objectIdA))
                jaccardScore = float(intersectingPixels) / float(unionPixels) 

                # append to object's score list
                scores.setdefault(globalIdA, []).append( (gtLabel, jaccardScore) )

                # store this as GT mapping if there was no better object for this GT label yet
                if jaccardScore > groundTruthMinJaccardScore and \
                    ((frame, gtLabel) not in gtToGlobalIdMap or gtToGlobalIdMap[(frame, gtLabel)][-1][1] < jaccardScore):
                    gtToGlobalIdMap.setdefault((frame, gtLabel), []).append((globalIdA, jaccardScore))

    # sort all gt mappings by ascending jaccard score
    for _, v in gtToGlobalIdMap.items():
        v.sort(key=lambda x: x[1]) 

    return frame, scores, gtToGlobalIdMap


class ConflictingSegmentsProbabilityGenerator(IlpProbabilityGenerator):
    """
    Specialization of the probability generator that computes all the features on its own,
    to have more than one segmentation hypotheses per timeframe.

    First step: make sure that objects from different hypotheses have different IDs

    * do that by adding the maxId of the "previous" segmentation hypothesis for that frame
    * store reference which hypothesis this segment comes from in Traxel, so that we can 
      reconstruct a result from the graph and images

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
        self._findOverlaps()

    def _findOverlaps(self):
        """
        Check which objects are overlapping between the different segmentation hypotheses,
        and store that information in every traxel.
        """
        getLogger().info("Checking for overlapping segmentation hypotheses...")
        t0 = time.time()

        # find exclusion constraints
        if self._useMultiprocessing:
            # use ProcessPoolExecutor, which instanciates as many processes as there CPU cores by default
            ExecutorType = concurrent.futures.ProcessPoolExecutor
            getLogger().info('Parallelizing via multiprocessing on all cores!')
        else:
            ExecutorType = DummyExecutor
            getLogger().info('Running on single core!')

        jobs = []
        progressBar = ProgressBar(stop=self.timeRange[1] - self.timeRange[0])
        progressBar.show(increase=0)

        with ExecutorType() as executor:
            for frame in range(self.timeRange[0], self.timeRange[1]):
                jobs.append(executor.submit(findConflictingHypothesesInSeparateProcess,
                                            frame,
                                            self._labelImageFilenames,
                                            self._labelImagePaths,
                                            self._labelImageFrameIdToGlobalId,
                                            self._pluginPaths
                ))
            for job in concurrent.futures.as_completed(jobs):
                progressBar.show()
                frame, overlaps = job.result()
                for objectId, overlapIds in overlaps.items():
                    if self.TraxelsPerFrame[frame][objectId].conflictingTraxelIds is None:
                        self.TraxelsPerFrame[frame][objectId].conflictingTraxelIds = []
                    self.TraxelsPerFrame[frame][objectId].conflictingTraxelIds.extend(overlapIds)
        
        t1 = time.time()
        getLogger().info("Finding overlaps took {} secs".format(t1 - t0))

    def findGroundTruthJaccardScoreAndMapping(self, 
                                              hypothesesGraph,
                                              groundTruthSegmentationFilename=None,
                                              groundTruthSegmentationPath=None,
                                              groundTruthTextFilename=None,
                                              groundTruthMinJaccardScore=0.5):
        """
        Find the overlap between all objects in the given segmentations with the groundtruth,
        and store that jaccard score in each traxel's features.

        **Returns** a solution dictionary in our JSON format, which fits to the given hypotheses graph.

        TODO: simplify this method! 
        Currently there are 4 different sets of IDs to reference nodes:
        * the ground truth trackId
        * a corresponding globalId (which is unique within a frame across different segmentation hypotheses)
        * an objectId (which equals the labelId within one segmentation hypotheses)
        * a globally unique UUID as used in the JSON files

        The nodes in the hypotheses graph are indexed by (frame, globalId), the resulting dict must use UUIDs.
        """

        getLogger().info("Computing Jaccard scores w.r.t. GroundTruth ...")
        t0 = time.time()

        # find exclusion constraints
        if self._useMultiprocessing:
            # use ProcessPoolExecutor, which instanciates as many processes as there CPU cores by default
            ExecutorType = concurrent.futures.ProcessPoolExecutor
            getLogger().info('Parallelizing via multiprocessing on all cores!')
        else:
            ExecutorType = DummyExecutor
            getLogger().info('Running on single core!')

        jobs = []
        progressBar = ProgressBar(stop=self.timeRange[1] - self.timeRange[0])
        progressBar.show(increase=0)
        gtFrameIdToGlobalIdsWithScoresMap = {}

        with ExecutorType() as executor:
            for frame in range(self.timeRange[0], self.timeRange[1]):
                jobs.append(executor.submit(computeJaccardScoresOnCloud,
                                            frame,
                                            self._labelImageFilenames,
                                            self._labelImagePaths,
                                            self._labelImageFrameIdToGlobalId,
                                            groundTruthSegmentationFilename,
                                            groundTruthSegmentationPath,
                                            groundTruthMinJaccardScore,
                                            self._pluginPaths
                ))
            for job in concurrent.futures.as_completed(jobs):
                progressBar.show()
                frame, scores, frameGtToGlobalIdMap = job.result()
                for objectId, individualScores in scores.items():
                    self.TraxelsPerFrame[frame][objectId].Features['JaccardScores'] = individualScores
                gtFrameIdToGlobalIdsWithScoresMap.update(frameGtToGlobalIdMap)
        
        t1 = time.time()
        getLogger().info("Finding jaccard scores took {} secs".format(t1 - t0))

        # create JSON result by mapping it to the hypotheses graph
        traxelIdPerTimestepToUniqueIdMap, _ = hypothesesGraph.getMappingsBetweenUUIDsAndTraxels()
        detectionResults = []
        for gtFrameAndId, globalIdsAndScores in gtFrameIdToGlobalIdsWithScoresMap.items():
            detectionResults.append({"id": traxelIdPerTimestepToUniqueIdMap[str(gtFrameAndId[0])][str(globalIdsAndScores[-1][0])], "value":1})
        
        # read tracks from textfile
        with open(groundTruthTextFilename, 'r') as tracksFile:
            lines = tracksFile.readlines()
        tracks = [[int(x) for x in line.strip().split(" ")] for line in lines]

        # order them by track start time and process track by track
        tracks.sort(key=lambda x: x[1])

        linkingResults = []
        descendants = {}
        missingLinks = 0

        def checkLinkExists(gtSrc, gtDest):
            # first check that both GT nodes have been mapped to a hypotheses
            if gtSrc in gtFrameIdToGlobalIdsWithScoresMap:
                src = (gtSrc[0], gtFrameIdToGlobalIdsWithScoresMap[gtSrc][-1][0])
            else:
                getLogger().warning("GT link's source node {} has no match in the segmentation hypotheses".format(gtSrc))
                return False

            if gtDest in gtFrameIdToGlobalIdsWithScoresMap:
                dest = (gtDest[0], gtFrameIdToGlobalIdsWithScoresMap[gtDest][-1][0])
            else:
                getLogger().warning("GT link's destination node {} has no match in the segmentation hypotheses".format(gtDest))
                return False
            
            # then map them to the hypotheses graph
            if not hypothesesGraph.hasNode(src):
                getLogger().warning("Source node of GT link {} was not found in graph".format((gtSrc, gtDest)))
                return False
            if not hypothesesGraph.hasNode(dest):
                getLogger().warning("Destination node of GTlink {} was not found in graph".format((gtSrc, gtDest)))
                return False
            if not hypothesesGraph.hasEdge(src, dest):
                getLogger().warning("Nodes are present, but GT link {} was not found in graph".format((gtSrc, gtDest)))
                return False
            return True

        def gtIdPerFrameToUuid(frame, gtId):
            return traxelIdPerTimestepToUniqueIdMap[str(frame)][str(gtFrameIdToGlobalIdsWithScoresMap[(frame, gtId)][-1][0])]

        # add links of all tracks
        for track in tracks:
            trackId, startFrame, endFrame, parent = track

            if parent != 0:
                descendants.setdefault(parent, []).append((startFrame, trackId))

            # add transitions along track
            for frame in range(startFrame, min(endFrame, self.timeRange[1])):
                if not checkLinkExists((frame, trackId), (frame + 1, trackId)):
                    getLogger().warning("Ignoring GT link from {} to {}".format((frame, trackId), (frame + 1, trackId)))
                    missingLinks += 1
                    continue

                link = {
                    "src":gtIdPerFrameToUuid(frame, trackId),
                    "dest":gtIdPerFrameToUuid(frame + 1, trackId),
                    "value":1
                }
                linkingResults.append(link)

        # construct divisions
        divisionResults = []
        for parent, childrenFrameIds in descendants.items():
            if len(childrenFrameIds) != 2:
                getLogger().warning("Found track {} that had descendants, but not exactly two. Ignoring it".format(parent))
                continue
            if childrenFrameIds[0][0] != childrenFrameIds[1][0]:
                getLogger().warning("Track {} divided, but children are not in same timeframe. Ignoring it".format(parent))
                continue

            # all good, found a proper division. Make sure the mother-daughter-links are available in the hypotheses graph 
            foundAllLinks = True
            divisionFrame = childrenFrameIds[0][0] - 1
            if divisionFrame >= self.timeRange[1]:
                continue
            for i in [0, 1]:
                foundAllLinks = foundAllLinks and checkLinkExists((divisionFrame, parent), (childrenFrameIds[i][0], childrenFrameIds[i][1]))

            if foundAllLinks:
                divisionResults.append({"id": gtIdPerFrameToUuid(divisionFrame, parent), "value": 1})
                for i in [0, 1]:
                    if not checkLinkExists((divisionFrame, parent), (childrenFrameIds[i][0], childrenFrameIds[i][1])):
                        getLogger().warning("Ignoring GT link from {} to {}".format((frame, trackId), (frame + 1, trackId)))
                        continue
                    link = {
                        "src":gtIdPerFrameToUuid(divisionFrame, parent),
                        "dest":gtIdPerFrameToUuid(childrenFrameIds[i][0], childrenFrameIds[i][1]),
                        "value":1
                    }
                    linkingResults.append(link)
            else:
                getLogger().warning("Division of {} ignored, could not find the links to the children, or not all participating GT nodes found a mapping".format(parent))
                missingLinks += 1

        getLogger().info("Ground Truth mapping could not find an equivalent for {} links, {} links projected.".format(missingLinks, len(linkingResults)))

        result = {}
        result['detectionResults'] = detectionResults
        result['linkingResults'] = linkingResults
        result['divisionResults'] = divisionResults
        return result


    def _insertFilenameAndIdToFeatures(self, featureDict, filename):
        """
        For later disambiguation, we store for each row in the feature matrix which file it came from.
        We also store the label image id.  
        """

        # get num elements:
        numElements = None

        for _, v in featureDict.items():
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
        for k, v in nextDict.items():
            assert(k in originalDict) # all frames should have the same features
            if isinstance(v, np.ndarray):
                originalDict[k] = np.concatenate((originalDict[k], v[1:]))
            else:
                originalDict[k].extend(v[1:])

    def _storeBackwardMapping(self, featuresPerFrame):
        """
        populates the `self._labelImageFrameIdToGlobalId` dictionary
        """
        for frame, featureDict in featuresPerFrame.items():
            for newId, (filename, objectId) in enumerate(zip(featureDict['filename'], featureDict['id'])):
                self._labelImageFrameIdToGlobalId[(filename, frame, objectId)] = newId

    def _extractAllFeatures(self, dispyNodeIps=[], turnOffFeatures=[]):
        """
        Extract the features of all frames of all segmentation hypotheses. 
        Feature extraction will be parallelized via multiprocessing.

        WARNING: distributed computation via Dispy is not supported here, so dispyNodeIps must be an empty list!
        """

        # configure progress bar
        numSteps = (self.timeRange[1] - self.timeRange[0]) * len(self._labelImageFilenames)
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
            # TODO: the division feature manager should also see the child candidates in all segmentation hypotheses
            for filename, path in zip(self._labelImageFilenames, self._labelImagePaths):
                if self._divisionClassifier is not None:
                    jobs = []
                    for frame in range(self.timeRange[0], self.timeRange[1] - 1):
                        jobs.append(executor.submit(computeDivisionFeaturesOnCloud,
                                                    frame,
                                                    featuresPerFrame[frame],
                                                    featuresPerFrame[frame + 1],
                                                    self._pluginManager.getImageProvider(),
                                                    filename,
                                                    path,
                                                    self.getNumDimensions(),
                                                    self._divisionFeatureNames
                        ))

                    for job in concurrent.futures.as_completed(jobs):
                        progressBar.show()
                        frame, feats = job.result()
                        # add division features to the dictionary for the first set, and then merge the new features in
                        if feats.keys()[0] not in featuresPerFrame[frame]:
                            featuresPerFrame[frame].update(feats)
                        else:
                            self._mergeFrameFeatures(featuresPerFrame[frame], feats)

        self._storeBackwardMapping(featuresPerFrame)

        t1 = time.time()
        getLogger().info("Feature computation took {} secs".format(t1 - t0))
        
        return featuresPerFrame