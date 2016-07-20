import numpy as np
from hytra.core.hypothesesgraph import HypothesesGraph, convertLegacyHypothesesGraphToJsonGraph, getTraxelFeatureVector


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
        return getTraxelFeatureVector(traxel, "detProb", max_state)


    def getDivisionFeatures(self, traxel):
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
