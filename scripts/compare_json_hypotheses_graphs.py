try:
    import commentjson as json
except ImportError:
    import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare two JSON graphs')
    parser.add_argument('--modelA', required=True, type=str, dest='modelFilenameA',
                        help='Filename of the json model description')
    parser.add_argument('--modelB', required=True, type=str, dest='modelFilenameB',
                        help='Filename of the second json model file')
    
    args = parser.parse_args()

    print("Loading model A: " + args.modelFilenameA)
    with open(args.modelFilenameA, 'r') as f:
        modelA = json.load(f)

    traxelIdPerTimestepToUniqueIdMap = modelA['traxelToUniqueId']
    timesteps = [t for t in traxelIdPerTimestepToUniqueIdMap.keys()]
    uuidToTraxelMapA = {}
    for t in timesteps:
        for i in traxelIdPerTimestepToUniqueIdMap[t].keys():
            uuid = traxelIdPerTimestepToUniqueIdMap[t][i]
            if uuid not in uuidToTraxelMapA:
                uuidToTraxelMapA[uuid] = []
            uuidToTraxelMapA[uuid].append((int(t), int(i)))

    print("Loading model B: " + args.modelFilenameB)
    with open(args.modelFilenameB, 'r') as f:
        modelB = json.load(f)

    traxelIdPerTimestepToUniqueIdMap = modelB['traxelToUniqueId']
    timesteps = [t for t in traxelIdPerTimestepToUniqueIdMap.keys()]
    uuidToTraxelMapB = {}
    for t in timesteps:
        for i in traxelIdPerTimestepToUniqueIdMap[t].keys():
            uuid = traxelIdPerTimestepToUniqueIdMap[t][i]
            if uuid not in uuidToTraxelMapB:
                uuidToTraxelMapB[uuid] = []
            uuidToTraxelMapB[uuid].append((int(t), int(i)))

    nodesA = set([obj['id'] for obj in modelA['segmentationHypotheses']])
    nodesB = set([obj['id'] for obj in modelB['segmentationHypotheses']])
    print("Size difference: len(A.nodes) - len(B.nodes) = {} - {} = {}".format(len(nodesA), len(nodesB), len(nodesA)-len(nodesB)))
    print("Nodes that differed: {}".format(nodesA ^ nodesB))

    nodeMapAtoB = {}
    nodeMapBtoA = {}
    for a in nodesA:
        trackletA = uuidToTraxelMapA[a]
        for b, trackletB in uuidToTraxelMapB.items():
            if trackletA == trackletB:
                assert b in nodesB
                nodeMapAtoB[a] = b
                nodeMapBtoA[b] = a

    linksA = set([(obj['src'], obj['dest']) for obj in modelA['linkingHypotheses']])
    linksAtransformed = set([(nodeMapAtoB[s], nodeMapAtoB[t]) for s,t in linksA])
    linksB = set([(obj['src'], obj['dest']) for obj in modelB['linkingHypotheses']])
    print("Size difference: len(A.links) - len(B.links) = {} - {} = {}".format(len(linksAtransformed), len(linksB), len(linksAtransformed)-len(linksB)))
    linkDiff = linksAtransformed ^ linksB
    print("Links that are not in both sets ({}):".format(len(linkDiff)))
    for a,b in linkDiff:
        print("\t {}: {} -> {} (in model A: {})".format((a,b), uuidToTraxelMapB[a], uuidToTraxelMapB[b], (nodeMapBtoA[a], nodeMapBtoA[b])))

        