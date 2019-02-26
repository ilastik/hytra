"""
Utilities that help with loading / saving as well as constructing and parsing
hypotheses graphs stored in our json (or python dictionary) format.
"""
import copy
import logging
import numpy as np

try:
    import commentjson as json
except ImportError:
    import json
from hytra.util.progressbar import ProgressBar
from hytra.util.progressbar import DefaultProgressVisitor

# ----------------------------------------------------------------------------
# Utility functions

logger = logging.getLogger(__name__)


def readFromJSON(filename):
    """ Read a dictionary from JSON """
    with open(filename, "r") as f:
        return json.load(f)


def writeToFormattedJSON(filename, dictionary):
    """ Write a dictionary to JSON, but use proper readable formatting  """
    with open(filename, "w") as f:
        json.dump(dictionary, f, indent=4, separators=(",", ": "))


def getMappingsBetweenUUIDsAndTraxels(model):
    """
    From a dictionary encoded model, load the "traxelToUniqueId" mapping,
    create a reverse mapping, and return both.
    """

    # create reverse mapping from json uuid to (timestep,ID)
    traxelIdPerTimestepToUniqueIdMap = model["traxelToUniqueId"]
    timesteps = [t for t in traxelIdPerTimestepToUniqueIdMap.keys()]
    uuidToTraxelMap = {}
    for t in timesteps:
        for i in traxelIdPerTimestepToUniqueIdMap[t].keys():
            uuid = traxelIdPerTimestepToUniqueIdMap[t][i]
            if uuid not in uuidToTraxelMap:
                uuidToTraxelMap[uuid] = []
            uuidToTraxelMap[uuid].append((int(t), int(i)))

    # sort the list of traxels per UUID by their timesteps
    for v in uuidToTraxelMap.values():
        v.sort(key=lambda timestepIdTuple: timestepIdTuple[0])

    return traxelIdPerTimestepToUniqueIdMap, uuidToTraxelMap


def getMergersDetectionsLinksDivisions(result, uuidToTraxelMap):
    # load results and map indices
    mergers = [
        timestepIdTuple + (entry["value"],)
        for entry in result["detectionResults"]
        if entry["value"] > 1
        for timestepIdTuple in uuidToTraxelMap[int(entry["id"])]
    ]
    detections = [
        timestepIdTuple
        for entry in result["detectionResults"]
        if entry["value"] > 0
        for timestepIdTuple in uuidToTraxelMap[int(entry["id"])]
    ]
    if "divisionResults" in result and result["divisionResults"] is not None:
        divisions = [
            uuidToTraxelMap[int(entry["id"])][-1]
            for entry in result["divisionResults"]
            if entry["value"] == True
        ]
    else:
        divisions = None
    links = [
        (uuidToTraxelMap[int(entry["src"])][-1], uuidToTraxelMap[int(entry["dest"])][0])
        for entry in result["linkingResults"]
        if entry["value"] > 0
    ]

    # add all internal links of tracklets
    for v in uuidToTraxelMap.values():
        prev = None
        for timestepIdTuple in v:
            if prev is not None:
                links.append((prev, timestepIdTuple))
            prev = timestepIdTuple

    return mergers, detections, links, divisions


def getMergersPerTimestep(mergers, timesteps):
    """ returns mergersPerTimestep = { "<timestep>": {<idx>: <count>, <idx>: <count>, ...}, "<timestep>": {...}, ... } """

    """
    TODO: We're storing all the mergers in a dict in order to increase speed at the expense of efficiency. 
    This could certainly be done faster and memory-efficient if we just return the merger dict for all times, without using the timestep loop.
    """

    mergersDict = {}
    for time, tid, count in mergers:
        time = str(time)
        if time in mergersDict:
            mergersDict[time][tid] = count
        else:
            mergersDict[time] = {}
            mergersDict[time][tid] = count

    mergersPerTimestep = {}
    for time in timesteps:
        if time in mergersDict:
            mergersPerTimestep[time] = mergersDict[time]
        else:
            mergersPerTimestep[time] = {}

    return mergersPerTimestep


def getDetectionsPerTimestep(detections, timesteps):
    """ returns detectionsPerTimestep = { "<timestep>": [<idx>, <idx>, ...], "<timestep>": [...], ...} """

    """
    TODO: We're storing all the detections in a dict in order to increase speed at the expense of efficiency. 
    This could certainly be done faster and memory-efficient if we just return the detections dict for all times, without using the timestep loop.
    """

    detectionsDict = {}
    for time, id in detections:
        time = str(time)
        if time in detectionsDict:
            detectionsDict[time].append(id)
        else:
            detectionsDict[time] = [id]

    detectionsPerTimestep = {}
    for time in timesteps:
        if time in detectionsDict:
            detectionsPerTimestep[time] = detectionsDict[time]
        else:
            detectionsPerTimestep[time] = []

    return detectionsPerTimestep


def getLinksPerTimestep(links, timesteps):
    """ returns linksPerTimestep = { "<timestep>": [(<idxA> (at previous timestep), <idxB> (at timestep)), (<idxA>, <idxB>), ...], ...} """

    """
    TODO: We're storing all the links in a dict in order to increase speed at the expense of efficiency. 
    This could certainly be done faster and memory-efficient if we just return the links dict for all times, without using the timestep loop.
    """

    linksDict = {}
    for source, target in links:
        time = str(target[0])
        if time in linksDict:
            linksDict[time].append((source[1], target[1]))
        else:
            linksDict[time] = [(source[1], target[1])]

    linksPerTimestep = {}
    for time in timesteps:
        if time in linksDict:
            linksPerTimestep[time] = linksDict[time]
        else:
            linksPerTimestep[time] = []

    return linksPerTimestep


def getMergerLinks(linksPerTimestep, mergersPerTimestep, timesteps):
    """ returns merger links as triplets [("timestep", (sourceIdAtTMinus1, destIdAtT)), (), ...]"""
    # filter links: at least one of the two incident nodes must be a merger
    # for it to be added to the merger resolving graph
    mergerLinks = [
        (t, (a, b))
        for t in timesteps
        for a, b in linksPerTimestep[t]
        if a in mergersPerTimestep[str(int(t) - 1)] or b in mergersPerTimestep[t]
    ]
    return mergerLinks


def getDivisionsPerTimestep(divisions, linksPerTimestep, timesteps):
    """ returns divisionsPerTimestep = { "<timestep>": {<parentIdx>: [<childIdx>, <childIdx>], ...}, "<timestep>": {...}, ... } """
    if divisions is not None:
        # find children of divisions by looking for the active links
        divisionsPerTimestep = {}
        for t in timesteps:
            divisionsPerTimestep[t] = {}
            for div_timestep, div_idx in divisions:
                if div_timestep == int(t) - 1:
                    # we have an active division of the mother cell "div_idx" in the previous frame
                    children = [b for a, b in linksPerTimestep[t] if a == div_idx]
                    assert (
                        len(children) == 2
                    ), "Expected two children of {}, but found {}".format(
                        (div_timestep, div_idx), children
                    )
                    divisionsPerTimestep[t][div_idx] = children
    else:
        divisionsPerTimestep = dict([(t, {}) for t in timesteps])

    return divisionsPerTimestep


def negLog(features):
    """ compute the (clamped) negative log of every entry in the list/array """
    fa = np.array(features)
    fa[fa < 0.0000000001] = 0.0000000001
    return list(np.log(fa) * -1.0)


def listify(l):
    """ put every element of the list in it's own list, and thus extends the depth of nested lists by one """
    return [[e] for e in l]


def delistify(l):
    """ take every element out of it's own list """
    return [e[0] for e in l]


def checkForConvexity(feats):
    """ check whether the given array of numbers is convex, meaning that the difference between consecutive numbers never decreases """
    grad = feats[1:] - feats[0:-1]
    for i in range(len(grad) - 1):
        assert grad[i + 1] > grad[i]


def convexify(listOfNumbers, eps):
    features = np.array(listOfNumbers)
    if features.shape[1] != 1:
        raise ValueError(
            "This script can only convexify feature vectors with one feature per state!"
        )

    # Note from Numpy Docs: In case of multiple occurrences of the minimum values, the indices corresponding to the first occurrence are returned.
    bestState = np.argmin(features)

    for direction in [-1, 1]:
        pos = bestState + direction
        previousGradient = 0
        while pos >= 0 and pos < features.shape[0]:
            newGradient = features[pos] - features[pos - direction]
            if np.abs(newGradient - previousGradient) < eps:
                # cost function's derivative is roughly constant, add epsilon
                previousGradient += eps
                features[pos] = features[pos - direction] + previousGradient
            elif newGradient < previousGradient:
                # cost function got too flat, set feature value to match old slope
                previousGradient += eps
                features[pos] = features[pos - direction] + previousGradient
            else:
                # all good, continue with new slope
                previousGradient = newGradient

            pos += direction
    try:
        checkForConvexity(features)
    except:
        logger.warning("Failed convexifying {}".format(features))
    return listify(features.flatten())


# ----------------------------------------------------------------------------
# helper class for graph-dictionaries


class JsonTrackingGraph(object):
    """
    Convenience class to handle a hypotheses graph stored as dictionary,
    which is transparently saved/loaded to JSON files.
    """

    def __init__(
        self,
        model=None,
        weights=None,
        result=None,
        model_filename=None,
        weights_filename=None,
        result_filename=None,
        progressVisitor=DefaultProgressVisitor(),
    ):

        assert weights is None or weights_filename is None
        assert model is None or model_filename is None
        assert result is None or result_filename is None

        # default values
        self.traxelIdPerTimestepToUniqueIdMap = {}
        if model is None:
            self.model = {
                "segmentationHypotheses": [],
                "linkingHypotheses": [],
                "exclusions": [],
                "divisionHypotheses": [],
                "traxelToUniqueId": self.traxelIdPerTimestepToUniqueIdMap,
                "settings": {
                    "statesShareWeights": True,
                    "allowPartialMergerAppearance": False,
                    "requireSeparateChildrenOfDivision": True,
                    "optimizerEpGap": 0.01,
                    "optimizerVerbose": True,
                    "optimizerNumThreads": 1,
                },
            }
        else:
            assert "segmentationHypotheses" in model
            assert "linkingHypotheses" in model
            self.model = model
        self.weights = weights
        self.result = result
        self.uuidToTraxelMap = {}

        # load from file if specified
        if model_filename is not None:
            logger.debug("Loading model file: " + model_filename)
            self.model = readFromJSON(model_filename)

        if weights_filename is not None:
            logger.debug("Loading weights file: " + weights_filename)
            self.weights = readFromJSON(weights_filename)

        if result_filename is not None:
            logger.debug("Loading result file: " + result_filename)
            self.result = readFromJSON(result_filename)

        # further initializations
        if model is not None or model_filename is not None:
            self.traxelIdPerTimestepToUniqueIdMap, self.uuidToTraxelMap = getMappingsBetweenUUIDsAndTraxels(
                self.model
            )

        self._nextUuid = 0

        self.progressVisitor = progressVisitor

    def addDetectionHypothesesFromTracklet(
        self,
        listOfTraxels,
        detectionFeatures,
        divisionFeatures=None,
        appearanceFeatures=None,
        disappearanceFeatures=None,
        **kwargs
    ):
        """
        Create a detection based on a `listOfTraxels` (because we can have tracklets). 
        Generates a new unique ID that represents this detection in the graph as one node and sets up the respective mappings
        in `JsonTrackingGraph.traxelIdPerTimestepToUniqueIdMap` and `JsonTrackingGraph.uuidToTraxelMap`.

        All further arguments in `**kwargs` are added to the detection dict in `segmentationHypotheses`.
        """
        assert listOfTraxels is not None and len(listOfTraxels) > 0

        # store mapping of all contained traxels to this detection uuid
        self.uuidToTraxelMap[self._nextUuid] = []
        for t in listOfTraxels:
            self.traxelIdPerTimestepToUniqueIdMap.setdefault(str(t.Timestep), {})[
                str(t.Id)
            ] = self._nextUuid
            self.uuidToTraxelMap[self._nextUuid].append((int(t.Timestep), int(t.Id)))

        return self.addDetectionHypotheses(
            detectionFeatures,
            divisionFeatures=divisionFeatures,
            appearanceFeatures=appearanceFeatures,
            disappearanceFeatures=disappearanceFeatures,
            **kwargs
        )

    def hasDivisions(self):
        """
        check all division and segmentation hypotheses whether there is any possible division present,
        because only then we need the division weight to be passed in. 
        """
        if (
            "divisionHypotheses" in self.model
            and len(self.model["divisionHypotheses"]) > 0
        ):
            return True

        for s in self.model["segmentationHypotheses"]:
            if "divisionFeatures" in s and len(s["divisionFeatures"]) > 0:
                return True
        return False

    def weightsListToDict(self, listOfWeights):
        """
        Given a list of 5 weights for `[transWeight, detWeight, divWeight, appearance_cost, disappearance_cost]`,
        **return** a weight dict that matches the current tracking model.
        """
        assert len(listOfWeights) == 5
        w = copy.deepcopy(listOfWeights)
        if not self.hasDivisions():
            del w[2]
        return {"weights": w}

    def weightsDictToList(self, weightsDict):
        """
        **return** a 5-element list for [transWeight, detWeight, divWeight, appearance_cost, disappearance_cost], given a dict of weights. 
        """
        assert "weights" in weightsDict
        w = copy.deepcopy(weightsDict["weights"])
        if not self.hasDivisions():
            w.insert(2, 0)
        return w

    def addDetectionHypotheses(self, features, **kwargs):
        """
        Construct a detection with the given features, assign it a new Uuid, and add it to the model's `segmentationHypotheses`.

        All further arguments in `**kwargs` are added to the detection dict in `segmentationHypotheses`.

        **Returns:** the unique ID of the newly created node in the graph
        """
        detection = {"id": self._nextUuid, "features": features}
        for k, v in kwargs.items():
            if v != None:
                detection[k] = v

        self.model["segmentationHypotheses"].append(detection)
        self._nextUuid += 1

        return detection["id"]

    def addLinkingHypotheses(self, srcUuid, destUuid, features, **kwargs):
        """
        Add a link to the JSON encoded graph between two nodes which are identified by their unique ids
        """
        link = {"src": srcUuid, "dest": destUuid, "features": features}
        for k, v in kwargs.items():
            link[k] = v

        self.model["linkingHypotheses"].append(link)

    def getNumDetections(self):
        return len(self.model["segmentationHypotheses"])

    def getNumLinks(self):
        return len(self.model["linkingHypotheses"])

    def convexifyCosts(self, epsilon=0.000001):
        """
        Convexify all cost vectors in this model (in place!).
        If two values are equal, the specified `epsilon` will be added to make sure the gradient
        does not stay at 0.

        Needed to run the flow solver afterwards
        """
        if not self.model["settings"]["statesShareWeights"]:
            raise ValueError(
                "This script can only convexify feature vectors with shared weights!"
            )

        if "segmentationHypotheses" in self.model:
            segmentationHypotheses = self.model["segmentationHypotheses"]
        else:
            segmentationHypotheses = []

        if "linkingHypotheses" in self.model:
            linkingHypotheses = self.model["linkingHypotheses"]
        else:
            linkingHypotheses = []

        if "divisionHypotheses" in self.model:
            divisionHypotheses = self.model["divisionHypotheses"]
        else:
            divisionHypotheses = []

        self.progressVisitor.showState("Convexify costs")
        numElements = (
            len(segmentationHypotheses)
            + len(linkingHypotheses)
            + len(divisionHypotheses)
        )
        countElements = 0
        for seg in segmentationHypotheses:
            countElements += 1
            self.progressVisitor.showProgress(countElements / float(numElements))
            for f in ["features", "appearanceFeatures", "disappearanceFeatures"]:
                if f in seg:
                    try:
                        seg[f] = convexify(seg[f], epsilon)
                    except:
                        logger.warning(
                            "Convexification failed for feature {} of :{}".format(
                                f, seg
                            )
                        )
                        exit(0)
            # division features are always convex (2 values defines just a line)

        for link in linkingHypotheses:
            countElements += 1
            self.progressVisitor.showProgress(countElements / float(numElements))
            link["features"] = convexify(link["features"], epsilon)

        for division in divisionHypotheses:
            countElements += 1
            self.progressVisitor.showProgress(countElements / float(numElements))
            division["features"] = convexify(division["features"], epsilon)

    def toHypothesesGraph(self):
        """
        From a json graph representation (and possibly a json result), 
        set up a hypotheses graph with the respective links.

        WARNING: only builds the structure of the graph at the moment, 
                 features/probabilities are not inserted!
        WARNING: builds the trackletgraph, not the full graph!
        """
        from hytra.core.hypothesesgraph import HypothesesGraph
        from hytra.core.probabilitygenerator import Traxel

        # set up graph
        hypothesesGraph = HypothesesGraph()
        for s in self.model["segmentationHypotheses"]:
            tracklet = self.uuidToTraxelMap[s["id"]]
            assert len(tracklet) > 0
            traxel = Traxel()
            traxel.Timestep = tracklet[0][0]
            traxel.Id = tracklet[0][1]
            hypothesesGraph.addNodeFromTraxel(traxel, tracklet=tracklet)
            # adding nodes automatically assigns UUIDs, we replace them by the loaded one
            hypothesesGraph._graph.node[(traxel.Timestep, traxel.Id)]["id"] = s["id"]

        # insert edges
        for l in self.model["linkingHypotheses"]:
            try:
                srcTracklet = self.uuidToTraxelMap[l["src"]]
                destTracklet = self.uuidToTraxelMap[l["dest"]]
            except:
                logger.warning(
                    "Failed finding {} from JSON['linkingHypotheses'] in uuidToTraxelMap".format(
                        (l["dest"], l["src"])
                    )
                )
            hypothesesGraph._graph.add_edge(
                (srcTracklet[0][0], srcTracklet[0][1]),
                ((destTracklet[0][0], destTracklet[0][1])),
            )

        # insert result
        if self.result is not None:
            hypothesesGraph.insertSolution(self.result)

        return hypothesesGraph

    def setTraxelToUniqueId(self, traxelIdPerTimestepToUniqueIdMap):
        """
        Set traxelToUniqueId map.
        """
        self.model["traxelToUniqueId"] = traxelIdPerTimestepToUniqueIdMap
