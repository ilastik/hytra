import hytra.core.jsongraph as jg


def return_example_model():
    model = {
        "linkingHypotheses": [
            {
                "dest": 4,
                "src": 0,
                "features": [
                    [0.00061062674861252415],
                    [7.4013299693657562],
                    [7.4013299693657562],
                ],
            },
            {
                "dest": 1,
                "src": 3,
                "features": [
                    [0.0039328296162961457],
                    [5.5403618783766868],
                    [5.5403618783766868],
                ],
            },
            {
                "dest": 2,
                "src": 3,
                "features": [
                    [9.8654350212315156e-05],
                    [9.2239375557990968],
                    [9.2239375557990968],
                ],
            },
            {"dest": 3, "src": 4, "features": [[23.025850929940457], [-0.0], [-0.0]]},
            {
                "dest": 4,
                "src": 5,
                "features": [
                    [7.426458619834081e-05],
                    [9.5079134845215343],
                    [9.5079134845215343],
                ],
            },
        ],
        "segmentationHypotheses": [
            {
                "appearanceFeatures": [[0.0], [0.0], [0.0]],
                "timestep": [0, 0],
                "id": 0,
                "disappearanceFeatures": [[0.0], [1.0], [1.0]],
                "features": [[23.025850929940457], [0.0], [23.025850929940457]],
            },
            {
                "appearanceFeatures": [[0.0], [1.0], [1.0]],
                "timestep": [3, 3],
                "id": 1,
                "disappearanceFeatures": [[0.0], [0.0], [0.0]],
                "features": [
                    [23.025850929940457],
                    [0.010050326220427584],
                    [4.6051702083398336],
                ],
            },
            {
                "appearanceFeatures": [[0.0], [1.0], [1.0]],
                "timestep": [3, 3],
                "id": 2,
                "disappearanceFeatures": [[0.0], [0.0], [0.0]],
                "features": [
                    [23.025850929940457],
                    [0.010050326220427584],
                    [4.6051702083398336],
                ],
            },
            {
                "appearanceFeatures": [[0.0], [1.0], [1.0]],
                "timestep": [2, 2],
                "id": 3,
                "disappearanceFeatures": [[0.0], [1.0], [1.0]],
                "features": [
                    [23.025850929940457],
                    [2.6592600326753035],
                    [0.072570685143913558],
                ],
            },
            {
                "appearanceFeatures": [[0.0], [1.0], [1.0]],
                "timestep": [1, 1],
                "id": 4,
                "disappearanceFeatures": [[0.0], [1.0], [1.0]],
                "features": [
                    [23.025850929940457],
                    [2.6592600326753035],
                    [0.072570685143913558],
                ],
            },
            {
                "appearanceFeatures": [[0.0], [0.0], [0.0]],
                "timestep": [0, 0],
                "id": 5,
                "disappearanceFeatures": [[0.0], [1.0], [1.0]],
                "features": [[23.025850929940457], [0.0], [23.025850929940457]],
            },
        ],
        "traxelToUniqueId": {
            "1": {"1": 4},
            "0": {"1": 0, "2": 5},
            "3": {"1": 2, "2": 1},
            "2": {"1": 3},
        },
        "exclusions": [],
        "divisionHypotheses": [],
        "settings": {
            "optimizerNumThreads": 1,
            "optimizerEpGap": 0.01,
            "requireSeparateChildrenOfDivision": True,
            "statesShareWeights": True,
            "optimizerVerbose": True,
            "allowPartialMergerAppearance": False,
        },
    }
    return model


def return_example_result():
    return {
        "detectionResults": [
            {"id": 0, "value": 1},
            {"id": 1, "value": 1},
            {"id": 2, "value": 1},
            {"id": 3, "value": 2},
            {"id": 4, "value": 2},
            {"id": 5, "value": 1},
        ],
        "divisionResults": None,
        "linkingResults": [
            {"dest": 4, "src": 0, "value": 1},
            {"dest": 1, "src": 3, "value": 1},
            {"dest": 2, "src": 3, "value": 1},
            {"dest": 3, "src": 4, "value": 2},
            {"dest": 4, "src": 5, "value": 1},
        ],
    }


def test_loading_no_divisions():
    model = return_example_model()
    result = return_example_result()

    # traxel <=> uuid mappings
    traxelIdPerTimestepToUniqueIdMap, uuidToTraxelMap = jg.getMappingsBetweenUUIDsAndTraxels(model)
    assert traxelIdPerTimestepToUniqueIdMap == {
        "0": {"1": 0, "2": 5},
        "1": {"1": 4},
        "2": {"1": 3},
        "3": {"1": 2, "2": 1},
    }
    assert uuidToTraxelMap == {
        0: [(0, 1)],
        1: [(3, 2)],
        2: [(3, 1)],
        3: [(2, 1)],
        4: [(1, 1)],
        5: [(0, 2)],
    }

    # get lists
    mergers, detections, links, divisions = jg.getMergersDetectionsLinksDivisions(result, uuidToTraxelMap)
    assert divisions is None
    assert mergers == [(2, 1, 2), (1, 1, 2)]
    assert detections == [(0, 1), (3, 2), (3, 1), (2, 1), (1, 1), (0, 2)]
    assert links == [
        ((0, 1), (1, 1)),
        ((2, 1), (3, 2)),
        ((2, 1), (3, 1)),
        ((1, 1), (2, 1)),
        ((0, 2), (1, 1)),
    ]

    # events per timestep
    timesteps = traxelIdPerTimestepToUniqueIdMap.keys()
    mergersPerTimestep = jg.getMergersPerTimestep(mergers, timesteps)
    assert mergersPerTimestep == {"0": {}, "1": {1: 2}, "2": {1: 2}, "3": {}}

    detectionsPerTimestep = jg.getDetectionsPerTimestep(detections, timesteps)
    assert detectionsPerTimestep == {"0": [1, 2], "1": [1], "2": [1], "3": [2, 1]}

    linksPerTimestep = jg.getLinksPerTimestep(links, timesteps)
    assert linksPerTimestep == {
        "0": [],
        "1": [(1, 1), (2, 1)],
        "2": [(1, 1)],
        "3": [(1, 2), (1, 1)],
    }

    # merger links as triplets [("timestep", (sourceId, destId)), (), ...]
    mergerLinks = jg.getMergerLinks(linksPerTimestep, mergersPerTimestep, timesteps)
    assert mergerLinks == [
        ("1", (1, 1)),
        ("1", (2, 1)),
        ("3", (1, 2)),
        ("3", (1, 1)),
        ("2", (1, 1)),
    ]


def test_toHypoGraph():
    model = return_example_model()
    result = return_example_result()
    trackingGraph = jg.JsonTrackingGraph(model=model, result=result)
    hypothesesGraph = trackingGraph.toHypothesesGraph()
    assert hypothesesGraph.countNodes() == 6
    assert hypothesesGraph.countArcs() == 5

    # traxel <=> uuid mappings
    traxelIdPerTimestepToUniqueIdMap, uuidToTraxelMap = hypothesesGraph.getMappingsBetweenUUIDsAndTraxels()
    assert traxelIdPerTimestepToUniqueIdMap == {
        "0": {"1": 0, "2": 5},
        "1": {"1": 4},
        "2": {"1": 3},
        "3": {"1": 2, "2": 1},
    }
    assert uuidToTraxelMap == {
        0: [(0, 1)],
        1: [(3, 2)],
        2: [(3, 1)],
        3: [(2, 1)],
        4: [(1, 1)],
        5: [(0, 2)],
    }

    for n in hypothesesGraph.nodeIterator():
        if n in [(1, 1), (2, 1)]:
            assert hypothesesGraph._graph.nodes[n]["value"] == 2
        else:
            assert hypothesesGraph._graph.nodes[n]["value"] == 1

    for a in hypothesesGraph.arcIterator():
        if a == ((1, 1), (2, 1)):
            assert hypothesesGraph._graph.edges[a[0], a[1]]["value"] == 2
        else:
            assert hypothesesGraph._graph.edges[a[0], a[1]]["value"] == 1


def test_weightListToFromDict():
    model = return_example_model()
    weights = [0, 1, 2, 3, 4]
    trackingGraph = jg.JsonTrackingGraph(model=model)
    wd = trackingGraph.weightsListToDict(weights)
    assert wd["weights"] == [0, 1, 3, 4]

    otherWeights = trackingGraph.weightsDictToList(wd)
    assert otherWeights == [0, 1, 0, 3, 4]
