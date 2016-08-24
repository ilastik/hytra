'''
Provide methods to find positive and negative training examples from a hypotheses graph and 
a ground truth mapping, in the presence of multiple competing segmentation hypotheseses.
'''

import numpy as np
import logging
from hytra.core.random_forest_classifier import RandomForestClassifier

def getLogger():
    ''' logger to be used in this module '''
    return logging.getLogger(__name__)

def trainDetectionClassifier(hypothesesGraph, gtFrameIdToGlobalIdsWithScoresMap, numSamples=100, selectedFeatures=None):
    """
    Finds the given number of training examples, half as positive and half as negative examples, from the
    given graph and mapping.

    Positive examples are those with the highest jaccard score, while negative examples can either 
    just not be the best match for a GT label, or also be not matched at all.



    **Returns**: a trained random forest
    """
    # create a list of all elements, sort them by their jaccard score, then pick from both ends?
    getLogger().debug("Extracting candidates")
    candidates = []

    nodeTraxelMap = hypothesesGraph.getNodeTraxelMap()
    for node in hypothesesGraph.nodeIterator():
        if 'JaccardScores' in nodeTraxelMap[node].Features and len(nodeTraxelMap[node].Features['JaccardScores']) > 0:
            globalIdsAndScores = nodeTraxelMap[node].Features['JaccardScores']
            globalIdsAndScores = sorted(globalIdsAndScores, key=lambda x: x[1])
            bestScore = globalIdsAndScores[-1][1]
            candidates.append( (node, bestScore) )
    
    assert(len(candidates) >= numSamples)
    candidates.sort(key=lambda x: x[1])

    # pick the first and last numSamples/2, and extract their features?
    # use RandomForestClassifier's method "extractFeatureVector"
    selectedSamples = candidates[0:numSamples//2] + candidates[-numSamples//2-1:-1]
    labels = np.hstack([np.zeros(numSamples//2), np.ones(numSamples//2)])

    # TODO: make sure that the positive examples were all selected in the GT mapping

    getLogger().debug("construct feature matrix")
    node = selectedSamples[0][0]
    if selectedFeatures is None:
        selectedFeatures = nodeTraxelMap[node].Features.keys()
        forbidden = ['JaccardScores', 'id', 'filename' , 'Polygon', 'detProb', 'divProb', 'com']
        forbidden += [f for f in selectedFeatures if f.count('_') > 0]
        for f in forbidden:
            if f in selectedFeatures:
                selectedFeatures.remove(f)
        getLogger().info("No list of selected features was specified, using {}".format(selectedFeatures))

    rf = RandomForestClassifier(selectedFeatures=selectedFeatures)
    features = rf.extractFeatureVector(nodeTraxelMap[node].Features, singleObject=True)
    featureMatrix = np.zeros( [len(selectedSamples), features.shape[1]] )
    featureMatrix[0, :] = features
    for idx, nodeAndScore in enumerate(selectedSamples[1:]):
        features = rf.extractFeatureVector(nodeTraxelMap[nodeAndScore[0]].Features, singleObject=True)
        featureMatrix[idx + 1, :] = features

    rf.train(featureMatrix, labels)

    return rf