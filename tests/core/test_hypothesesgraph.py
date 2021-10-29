import hytra.core.hypothesesgraph as hg
import hytra.core.probabilitygenerator as pg
import numpy as np
import networkx as nx
from hytra.core.probabilitygenerator import Traxel


def test_trackletgraph():
    h = hg.HypothesesGraph()
    nx.add_path(h._graph, [(0, 1), (1, 1), (2, 1), (3, 1)])
    for i in [(0, 1), (1, 1), (2, 1), (3, 1)]:
        t = Traxel()
        t.Timestep = i[0]
        t.Id = i[1]
        h._graph.nodes[i]["traxel"] = t

    t = h.generateTrackletGraph()
    assert t.countArcs() == 1
    assert t.countNodes() == 2
    assert "tracklet" in t._graph.nodes[(0, 1)]


def test_computeLineagesAndPrune():
    h = hg.HypothesesGraph()
    nx.add_path(h._graph, [(0, 0), (1, 1), (2, 2)])
    nx.add_path(h._graph, [(1, 1), (2, 3), (3, 4)])

    for n in h._graph.nodes:
        h._graph.nodes[n]["id"] = n[1]
        h._graph.nodes[n]["traxel"] = pg.Traxel()
        h._graph.nodes[n]["traxel"].Id = n[1]
        h._graph.nodes[n]["traxel"].Timestep = n[0]

    solutionDict = {
        "detectionResults": [
            {"id": 0, "value": 1},
            {"id": 1, "value": 1},
            {"id": 2, "value": 1},
            {"id": 3, "value": 1},
            {"id": 4, "value": 0},
        ],
        "linkingResults": [
            {"dest": 1, "src": 0, "value": 1},
            {"dest": 2, "src": 1, "value": 1},
            {"dest": 3, "src": 1, "value": 1},
            {"dest": 4, "src": 3, "value": 0},
        ],
        "divisionResults": [{"id": 1, "value": True}, {"id": 2, "value": False}],
    }

    h.insertSolution(solutionDict)
    h.computeLineage()
    h.pruneGraphToSolution(0)
    h.pruneGraphToSolution(1)


def test_computeLineagesWithMergers():
    h = hg.HypothesesGraph()
    nx.add_path(h._graph, [(0, 0), (1, 1), (2, 2)])
    nx.add_path(h._graph, [(0, 5), (1, 1), (2, 3), (3, 4)])

    for n in h._graph.nodes:
        h._graph.nodes[n]["id"] = n[1]
        h._graph.nodes[n]["traxel"] = pg.Traxel()
        h._graph.nodes[n]["traxel"].Id = n[1]
        h._graph.nodes[n]["traxel"].Timestep = n[0]

    solutionDict = {
        "detectionResults": [
            {"id": 0, "value": 1},
            {"id": 1, "value": 2},
            {"id": 2, "value": 1},
            {"id": 3, "value": 1},
            {"id": 4, "value": 1},
            {"id": 5, "value": 1},
        ],
        "linkingResults": [
            {"dest": 1, "src": 0, "value": 1},
            {"dest": 1, "src": 5, "value": 1},
            {"dest": 2, "src": 1, "value": 1},
            {"dest": 3, "src": 1, "value": 1},
            {"dest": 4, "src": 3, "value": 1},
        ],
        "divisionResults": [{"id": 1, "value": False}, {"id": 2, "value": False}],
    }

    h.insertSolution(solutionDict)
    h.computeLineage()

    assert h._graph.nodes[(0, 0)]["lineageId"] == 2
    assert h._graph.nodes[(0, 5)]["lineageId"] == 3
    assert h._graph.nodes[(1, 1)]["lineageId"] == 3
    assert h._graph.nodes[(1, 1)]["lineageId"] == 3
    assert h._graph.nodes[(2, 3)]["lineageId"] == 3
    assert h._graph.nodes[(3, 4)]["lineageId"] == 3


def test_insertAndExtractSolution():
    h = hg.HypothesesGraph()
    nx.add_path(h._graph, [(0, 0), (1, 1), (2, 2)])
    nx.add_path(h._graph, [(1, 1), (2, 3), (3, 4)])

    for n in h._graph.nodes:
        h._graph.nodes[n]["id"] = n[1]
        h._graph.nodes[n]["traxel"] = pg.Traxel()
        h._graph.nodes[n]["traxel"].Id = n[1]
        h._graph.nodes[n]["traxel"].Timestep = n[0]

    solutionDict = {
        "detectionResults": [
            {"id": 0, "value": 1},
            {"id": 1, "value": 1},
            {"id": 2, "value": 1},
            {"id": 3, "value": 1},
            {"id": 4, "value": 0},
        ],
        "linkingResults": [
            {"dest": 1, "src": 0, "value": 1},
            {"dest": 2, "src": 1, "value": 1},
            {"dest": 3, "src": 1, "value": 1},
            {"dest": 4, "src": 3, "value": 0},
        ],
        "divisionResults": [{"id": 1, "value": True}, {"id": 2, "value": False}],
    }

    h.insertSolution(solutionDict)
    outSolutionDict = h.getSolutionDictionary()

    # from solution to outSolution
    for group in ["detectionResults", "divisionResults"]:
        for entry in solutionDict[group]:
            ref = [m for m in outSolutionDict[group] if m["id"] == entry["id"]]
            assert len(ref) <= 1
            if len(ref) == 1:
                for k, v in ref[0].items():
                    assert v == entry[k]
            else:
                assert entry["value"] == 0

    # from outSolution to Solution
    for group in ["detectionResults", "divisionResults"]:
        for entry in outSolutionDict[group]:
            ref = [m for m in solutionDict[group] if m["id"] == entry["id"]]
            assert len(ref) <= 1
            if len(ref) == 1:
                for k, v in ref[0].items():
                    assert v == entry[k]
            else:
                assert entry["value"] == 0

    assert h._graph.nodes[(1, 1)]["divisionValue"] == 1
    assert h._graph.nodes[(2, 2)]["divisionValue"] == 0
    assert h._graph.nodes[(0, 0)]["value"] == 1
    assert h._graph.nodes[(1, 1)]["value"] == 1
    assert h._graph.nodes[(2, 2)]["value"] == 1
    assert h._graph.nodes[(2, 3)]["value"] == 1
    assert h._graph.nodes[(3, 4)]["value"] == 0
    assert h._graph.edges[(0, 0), (1, 1)]["value"] == 1
    assert h._graph.edges[(1, 1), (2, 2)]["value"] == 1
    assert h._graph.edges[(1, 1), (2, 3)]["value"] == 1
    assert h._graph.edges[(2, 3), (3, 4)]["value"] == 0

    h.computeLineage()
    assert set(h._graph.nodes[(1, 1)]["children"]) == set([(2, 2), (2, 3)])
    assert h._graph.nodes[(2, 2)]["parent"] == (1, 1)
    assert h._graph.nodes[(2, 3)]["parent"] == (1, 1)


def test_insertEnergies():
    skipLinkBias = 20
    h = hg.HypothesesGraph()
    nx.add_path(h._graph, [(0, 1), (1, 1), (2, 1), (3, 1)])
    for uuid, i in enumerate([(0, 1), (1, 1), (2, 1), (3, 1)]):
        t = Traxel()
        t.Timestep = i[0]
        t.Id = i[1]
        # fill in detProb, divProb, and center of mass
        t.Features["detProb"] = [0.2, 0.8]
        t.Features["divProb"] = [0.2, 0.8]
        t.Features["com"] = [float(i[0]), 0.0]

        h._graph.nodes[i]["traxel"] = t
        h._graph.nodes[i]["id"] = uuid

    # set up some dummy functions to compute probabilities from a traxel
    def detProbFunc(traxel):
        return traxel.Features["detProb"]

    def divProbFunc(traxel):
        return traxel.Features["divProb"]

    def boundaryCostFunc(traxel, forAppearance):
        return 1.0

    def transProbFunc(traxelA, traxelB):
        dist = np.linalg.norm(np.array(traxelA.Features["com"]) - np.array(traxelB.Features["com"]))
        return [1.0 - np.exp(-dist), np.exp(-dist)]

    h.insertEnergies(1, detProbFunc, transProbFunc, boundaryCostFunc, divProbFunc, skipLinkBias)

    for n in h.nodeIterator():
        assert "features" in h._graph.nodes[n]
        assert h._graph.nodes[n]["features"] == [
            [1.6094379124341003],
            [0.22314355131420971],
        ]
        assert "divisionFeatures" in h._graph.nodes[n]
        assert h._graph.nodes[n]["divisionFeatures"] == [
            [1.6094379124341003],
            [0.22314355131420971],
        ]
        assert "appearanceFeatures" in h._graph.nodes[n]
        assert h._graph.nodes[n]["appearanceFeatures"] == [[0.0], [1.0]]
        assert "disappearanceFeatures" in h._graph.nodes[n]
        assert h._graph.nodes[n]["disappearanceFeatures"] == [[0.0], [1.0]]

    for a in h.arcIterator():
        assert "features" in h._graph.edges[a[0], a[1]]
        srcTraxel = h._graph.nodes[h.source(a)]["traxel"]
        destTraxel = h._graph.nodes[h.target(a)]["traxel"]
        frame_gap = destTraxel.Timestep - srcTraxel.Timestep
        assert h._graph.edges[a[0], a[1]]["features"] == [
            [0.45867514538708193],
            [1.0 + skipLinkBias * (frame_gap - 1)],
        ]


if __name__ == "__main__":
    test_trackletgraph()
    test_insertAndExtractSolution()
    test_computeLineagesAndPrune()
    test_computeLineagesWithMergers()
    test_insertEnergies()
