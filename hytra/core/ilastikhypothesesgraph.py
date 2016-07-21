import logging
import numpy as np
from hytra.core.hypothesesgraph import HypothesesGraph, getTraxelFeatureVector, negLog, listify
import hytra.core.jsongraph
from hytra.util.progressbar import ProgressBar

def getLogger():
    ''' logger to be used in this module '''
    return logging.getLogger(__name__)

class IlastikHypothesesGraph(HypothesesGraph):
    '''
    Hypotheses graph specialized for the ConservationTracking implementation in ilastik.
    '''

    def __init__(self, 
                 probabilityGenerator,
                 timeRange, 
                 maxNumObjects, 
                 numNearestNeighbors,
                 fieldOfView,
                 divisionThreshold=0.1,
                 withDivisions=True,
                 borderAwareWidth=10,
                 maxNeighborDistance=200,
                 transitionParameter=5.0,
                 transitionClassifier=None):
        '''
        Constructor
        '''
        super(IlastikHypothesesGraph, self).__init__()

        # store values
        self.probabilityGenerator = probabilityGenerator
        self.timeRange = timeRange
        self.maxNumObjects = maxNumObjects
        self.numNearestNeighbors = numNearestNeighbors
        self.fieldOfView = fieldOfView
        self.divisionThreshold = divisionThreshold
        self.withDivisions = withDivisions
        self.borderAwareWidth = borderAwareWidth
        self.maxNeighborDistance = maxNeighborDistance
        self.transitionClassifier = transitionClassifier
        self.transitionParameter = transitionParameter

        # build hypotheses graph
        self.buildFromProbabilityGenerator(probabilityGenerator,
                                           numNearestNeighbors=numNearestNeighbors,
                                           maxNeighborDist=maxNeighborDistance,
                                           withDivisions=withDivisions,
                                           divisionThreshold=0.1)

    def insertEnergies(self):
        """
        Inserts the energies (AKA features) into the graph, such that each node and link 
        hold all information needed to run tracking.

        See the documentation of `hytra.core.hypothesesgraph` for details on how the features are stored.
        """
        # define wrapper functions
        def detectionProbabilityFunc(traxel):
            return self.getDetectionFeatures(traxel, self.maxNumObjects + 1)

        def transitionProbabilityFunc(srcTraxel, destTraxel):
            if self.transitionClassifier is None:
                return self.getTransitionFeaturesDist(srcTraxel, destTraxel, self.transitionParameter, self.maxNumObjects + 1)
            else:
                return self.getTransitionFeaturesRF(srcTraxel, destTraxel, self.transitionClassifier, self.probabilityGenerator, self.maxNumObjects + 1)

        def boundaryCostMultiplierFunc(traxel):
            return self.getBoundaryCostMultiplier(traxel, self.fieldOfView, self.borderAwareWidth, self.timeRange[0], self.timeRange[-1])

        def divisionProbabilityFunc(traxel):
            try:
                divisionFeatures = self.getDivisionFeatures(traxel)
                if divisionFeatures[0] > self.divisionThreshold:
                    divisionFeatures = list(reversed(divisionFeatures))
                else:
                    divisionFeatures = None
            except:
                divisionFeatures = None
            return divisionFeatures

        super(IlastikHypothesesGraph, self).insertEnergies(
            self.maxNumObjects,
            detectionProbabilityFunc,
            transitionProbabilityFunc,
            boundaryCostMultiplierFunc,
            divisionProbabilityFunc)

    def getDetectionFeatures(self, traxel, max_state):
        """
        USe the detection probabilities stored as `detProb` in the features of the traxel
        """
        return getTraxelFeatureVector(traxel, "detProb", max_state)


    def getDivisionFeatures(self, traxel):
        """
        Use the division probability stored in the features of the given traxel.
        """
        prob = traxel.get_feature_value("divProb", 0)
        return [1.0 - prob, prob]


    def getTransitionFeaturesDist(self, traxelA, traxelB, transitionParam, max_state):
        """
        Get the transition probabilities based on the object's distance
        """
        positions = [np.array([t.X(), t.Y(), t.Z()]) for t in [traxelA, traxelB]]
        dist = np.linalg.norm(positions[0] - positions[1])
        prob = np.exp(-dist / transitionParam)
        return [1.0 - prob] + [prob] * (max_state - 1)


    def getTransitionFeaturesRF(self, traxelA, traxelB, transitionClassifier, probabilityGenerator, max_state):
        """
        Get the transition probabilities by predicting them with the classifier
        """
        feats = [probabilityGenerator.getTraxelFeatureDict(obj.Timestep, obj.Id) for obj in [traxelA, traxelB]]
        featVec = probabilityGenerator.getTransitionFeatureVector(feats[0], feats[1], transitionClassifier.selectedFeatures)
        probs = transitionClassifier.predictProbabilities(featVec)[0]
        return [probs[0]] + [probs[1]] * (max_state - 1)


    def getBoundaryCostMultiplier(self, traxel, fov, margin, t0, t1):
        """
        A traxel's appearance and disappearance probability decrease linearly within a `margin` to the image border
        which is defined by the field of view `fov`. 
        Traxels in the first frame appear for free, and traxels in the last frame disappear for free.
        """
        if traxel.Timestep <= t0 or traxel.Timestep >= t1 - 1:
            return 0.0

        dist = fov.spatial_distance_to_border(traxel.Timestep, traxel.X(), traxel.Y(), traxel.Z(), False)
        if dist > margin:
            return 1.0
        else:
            if margin > 0:
                return float(dist) / margin
            else:
                return 1.0


def convertLegacyHypothesesGraphToJsonGraph(hypothesesGraph,
                                      nodeIterator,
                                      arcIterator,
                                      withTracklets,
                                      maxNumObjects,
                                      numElements,
                                      traxelMap,
                                      detectionProbabilityFunc,
                                      transitionProbabilityFunc,
                                      boundaryCostMultiplierFunc,
                                      divisionProbabilityFunc):
    '''
    Build a json representation of this hypotheses graph, by transforming the probabilities for certain
    events (given by the `*ProbabilityFunc`-functions per traxel) into energies. If the given graph
    contained tracklets (`withTracklets`), then also the probabilities over all contained traxels will be
    accumulated for those nodes in the graph.

    The `hypothesesGraph` as well as `nodeIterator` and `arcIterator` are needed as parameters to
    support the legacy pgmlink-style hypotheses graph as well.

    ** Parameters: **

    * `hypothesesGraph`: graph whose nodes and edges we are about to traverse.
    * `nodeIterator`: node iterator
    * `arcIterator`: arc iterator
    * `withTracklets`: whether tracklets are used
    * `maxNumObjects`: the max number of objects per detections
    * `numElements`: number of nodes + number of edges (for progress bar)
    * `traxelMap`: mapping from graph-node to list of traxels (in a tracklet)
    * `detectionProbabilityFunc`: should take a traxel and return its detection probabilities
     ([prob0objects, prob1object,...])
    * `transitionProbabilityFunc`: should take two traxels and return this link's probabilities
     ([prob0objectsInTransition, prob1objectsInTransition,...])
    * `boundaryCostMultiplierFunc`: should take a traxel and return a scalar multiplier between 0 and 1 for the
     appearance/disappearance cost that depends on the traxel's distance to the spacial and time boundary
    * `divisionProbabilityFunc`: should take a traxel and return its division probabilities
     ([probNoDiv, probDiv])
    '''

    ## TODO: store probabilities inside the networkx-graph and then 
    ## use the following networkx functionality to make this much nicer:
    # a = json_graph.node_link_data(g, attrs={'source':'src', 'target':'dest', 'id':'id', 'key':'test'})
    # a['segmentationHypotheses'] = a['nodes']
    # del a['nodes']
    # a['linkingHypotheses'] = a['links']
    # del a['nodes']

    getLogger().info("Creating JSON graph from legacy hypotheses graph")
    progressBar = ProgressBar(stop=numElements)
    trackingGraph = hytra.core.jsongraph.JsonTrackingGraph()

    # add all detections to JSON
    for n in nodeIterator:
        if not withTracklets:
            # only one traxel, but make it a list so everything below works the same
            traxels = [traxelMap[n]]
        else:
            traxels = traxelMap[n]

        # accumulate features over all contained traxels
        previousTraxel = None
        detectionFeatures = np.zeros(maxNumObjects + 1)
        for t in traxels:
            detectionFeatures += np.array(negLog(detectionProbabilityFunc(t)))
            if previousTraxel is not None:
                detectionFeatures += np.array(negLog(transitionProbabilityFunc(previousTraxel, t)))
            previousTraxel = t

        detectionFeatures = listify(list(detectionFeatures))

        # division only if probability is big enough
        divisionFeatures = divisionProbabilityFunc(traxels[-1])
        if divisionFeatures is not None:
            divisionFeatures = listify(negLog(divisionFeatures))

        # appearance/disappearance
        appearanceFeatures = listify([0.0] + [boundaryCostMultiplierFunc(traxels[0])] * maxNumObjects)
        disappearanceFeatures = listify([0.0] + [boundaryCostMultiplierFunc(traxels[-1])] * maxNumObjects)

        trackingGraph.addDetectionHypothesesFromTracklet(traxels,
                                                         detectionFeatures,
                                                         divisionFeatures,
                                                         appearanceFeatures,
                                                         disappearanceFeatures,
                                                         timestep=[traxels[0].Timestep, traxels[-1].Timestep])
        progressBar.show()

    # add all links
    for a in arcIterator:
        if not withTracklets:
            srcTraxel = traxelMap[hypothesesGraph.source(a)]
            destTraxel = traxelMap[hypothesesGraph.target(a)]
        else:
            srcTraxel = traxelMap[hypothesesGraph.source(a)][-1]  # src is last of the traxels in source tracklet
            destTraxel = traxelMap[hypothesesGraph.target(a)][0]  # dest is first of traxels in destination tracklet
        src = trackingGraph.traxelIdPerTimestepToUniqueIdMap[str(srcTraxel.Timestep)][str(srcTraxel.Id)]
        dest = trackingGraph.traxelIdPerTimestepToUniqueIdMap[str(destTraxel.Timestep)][str(destTraxel.Id)]

        features = listify(negLog(transitionProbabilityFunc(srcTraxel, destTraxel)))
        trackingGraph.addLinkingHypotheses(src, dest, features)
        progressBar.show()

    return trackingGraph