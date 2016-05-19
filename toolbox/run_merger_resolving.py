import commentjson as json
import logging
import os
import configargparse as argparse
import numpy as np
import h5py
import itertools
import networkx as nx
from pluginsystem.plugin_manager import TrackingPluginManager
import core.traxelstore as traxelstore
import core.jsongraph

def createUnresolvedGraph(divisionsPerTimestep, mergersPerTimestep, mergerLinks):
    """ set up a networkx graph consisting of mergers (not resolved yet!) and their direct neighbors """

    unresolvedGraph = nx.DiGraph()
    def source(timestep, link):
        return int(timestep) - 1, link[0]

    def target(timestep, link):
        return int(timestep), link[1]

    def addNode(node):
        ''' add a node to the unresolved graph and fill in the properties `division` and `count` '''
        intT, idx = node
        if divisionsPerTimestep is not None:
            division = idx in divisionsPerTimestep[str(intT)]
        else:
            division = False
        count = 1
        if idx in mergersPerTimestep[str(intT)]:
            assert(not division)
            count = mergersPerTimestep[str(intT)][idx]
        unresolvedGraph.add_node(node, division=division, count=count)

    # add nodes
    for t, link in mergerLinks:
        for n in [source(t, link), target(t, link)]:
            if not unresolvedGraph.has_node(n):
                addNode(n)
        unresolvedGraph.add_edge(source(t, link), target(t,link))

    return unresolvedGraph

def prepareResolvedGraph(unresolvedGraph): 
    """ 
    split up the division nodes if they have two outgoing links 
    and add the duplicates to the resolved graph 
    """
    resolvedGraph = unresolvedGraph.copy()
    numDivisionNodes = 0
    for n in unresolvedGraph:
        if unresolvedGraph.node[n]['division'] and len(unresolvedGraph.out_edges(n)) == 2:
            # create a duplicate node, make one link start from there
            duplicate = (n[0], 'div-{}'.format(numDivisionNodes))
            numDivisionNodes += 1
            resolvedGraph.add_node(duplicate, division=False, count=1)

            dest = unresolvedGraph.out_edges(n)[0]
            resolvedGraph.add_edge(duplicate, dest)
            resolvedGraph.remove_edge(n, dest)

            # store node references
            resolvedGraph.node[duplicate]['origin'] = n
            resolvedGraph.node[n]['duplicate'] = duplicate

    return resolvedGraph

def refineMergerSegmentations(resolvedGraph, 
                                unresolvedGraph, 
                                timesteps, 
                                pluginManager, 
                                labelImageFilename, 
                                labelImagePath):
    ''' update segmentation of mergers from first timeframe to last and create new nodes: '''

    imageProvider = pluginManager.getImageProvider()
    mergerResolver = pluginManager.getMergerResolver()

    intTimesteps = [int(t) for t in timesteps]
    intTimesteps.sort()

    for intT in intTimesteps:
        t = str(intT)
        # use image provider plugin to load labelimage
        labelImage = imageProvider.getLabelImageForFrame(
            args.label_image_filename, args.label_image_path, int(t))
        labelImages[t] = labelImage
        nextObjectId = labelImage.max() + 1

        for idx in detectionsPerTimestep[t]:
            node = (intT, idx)
            if node not in resolvedGraph:
                continue

            count = 1
            if idx in mergersPerTimestep[t]:
                count = mergersPerTimestep[t][idx]
            print("Looking at node {} in timestep {} with count {}".format(idx, t, count))
            
            # collect initializations from incoming
            initializations = []
            for predecessor, _ in unresolvedGraph.in_edges(node):
                initializations.extend(unresolvedGraph.node[predecessor]['fits'])
            # TODO: what shall we do if e.g. a 2-merger and a single object merge to 2 + 1, 
            # so there are 3 initializations for the 2-merger, and two initializations for the 1 merger?
            # What does pgmlink do in that case?

            # use merger resolving plugin to fit `count` objects, also updates labelimage!
            fittedObjects = mergerResolver.resolveMerger(labelImage, idx, nextObjectId, count, initializations)
            assert(len(fittedObjects) == count)

            # split up node if count > 1, duplicate incoming and outgoing arcs
            if count > 1:
                for idx, fit in zip(range(nextObjectId, nextObjectId + count), fittedObjects):
                    newNode = (intT, idx)
                    resolvedGraph.add_node(newNode, division=False, count=1, origin=node)
                    
                    for e in unresolvedGraph.out_edges(node):
                        resolvedGraph.add_edge(newNode, e[1])
                    for e in unresolvedGraph.in_edges(node):
                        if 'newIds' in unresolvedGraph.node[e[0]]:
                            for newId in unresolvedGraph.node[e[0]]['newIds']:
                                resolvedGraph.add_edge((e[0][0], newId), newNode)
                        else:
                            resolvedGraph.add_edge(e[0], newNode)

                resolvedGraph.remove_node(node)
                unresolvedGraph.node[node]['newIds'] = range(nextObjectId, nextObjectId + count)
                nextObjectId += count

            # each unresolved node stores its fitted shape(s) to be used 
            # as initialization in the next frame, this way division duplicates 
            # and de-merged nodes in the resolved graph do not need to store a fit as well
            unresolvedGraph.node[node]['fits'] = fittedObjects

    # import matplotlib.pyplot as plt
    # nx.draw_networkx(resolvedGraph)
    # plt.savefig("/Users/chaubold/test.pdf")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Given a hypotheses json graph and a result.json, this script'
                        + ' resolves all mergers by updating the segmentation and inserting the appropriate '
                        + 'nodes and links.',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', is_config_file=True, help='config file path', dest='config_file', required=True)

    parser.add_argument('--graph-json-file', required=True, type=str, dest='model_filename',
                        help='IN Filename of the json model description')
    parser.add_argument('--result-json-file', required=True, type=str, dest='result_filename',
                        help='IN Filename of the json file containing results')
    parser.add_argument('--label-image-filename', required=True, type=str, dest='label_image_filename',
                        help='IN Filename of the original ilasitk tracking project')
    parser.add_argument('--label-image-path', dest='label_image_path', type=str,
                        default='/TrackingFeatureExtraction/LabelImage/0000/[[%d, 0, 0, 0, 0], [%d, %d, %d, %d, 1]]',
                        help='internal hdf5 path to label image')
    parser.add_argument('--raw-data-file', type=str, dest='raw_filename', default=None,
                      help='filename to the raw h5 file')
    parser.add_argument('--raw-data-path', type=str, dest='raw_path', default='volume/data',
                      help='Path inside the raw h5 file to the data')
    parser.add_argument('--transition-classifier-file', dest='transition_classifier_filename', type=str,
                        default=None)
    parser.add_argument('--transition-classifier-path', dest='transition_classifier_path', type=str, default='/')
    parser.add_argument('--out-model', type=str, dest='out_model_filename', required=True, 
                        help='Filename of the json model containing the hypotheses graph including new nodes')
    parser.add_argument('--out-label-image', type=str, dest='out_label_image', required=True, 
                        help='Filename where to store the label image with updated segmentation')
    parser.add_argument('--out-result', type=str, dest='out_result', required=True, 
                        help='Filename where to store the new result')
    parser.add_argument('--trans-par', dest='trans_par', type=float, default=5.0,
                        help='alpha for the transition prior')
    
    args, _ = parser.parse_known_args()
    logging.basicConfig(level=logging.INFO)

    logging.basicConfig(level=logging.DEBUG)

    with open(args.model_filename, 'r') as f:
        model = json.load(f)

    with open(args.result_filename, 'r') as f:
        result = json.load(f)
        assert(result['detectionResults'] is not None)
        assert(result['linkingResults'] is not None)
        withDivisions = result['divisionResults'] is not None

    traxelIdPerTimestepToUniqueIdMap, uuidToTraxelMap = core.jsongraph.getMappingsBetweenUUIDsAndTraxels(model)
    timesteps = [t for t in traxelIdPerTimestepToUniqueIdMap.keys()]
    mergers, detections, links, divisions = core.jsongraph.getMergersDetectionsLinksDivisions(result, uuidToTraxelMap, withDivisions)

    mergersPerTimestep = core.jsongraph.getMergersPerTimestep(mergers, timesteps)
    linksPerTimestep = core.jsongraph.getLinksPerTimestep(links, timesteps)
    detectionsPerTimestep = core.jsongraph.getDetectionsPerTimestep(detections, timesteps)
    divisionsPerTimestep = core.jsongraph.getDivisionsPerTimestep(divisions, linksPerTimestep, timesteps, withDivisions)
    mergerLinks = core.jsongraph.getMergerLinks(linksPerTimestep, mergersPerTimestep, timesteps)

    # ------------------------------------------------------------
    
    pluginManager = TrackingPluginManager(verbose=True)
    pluginManager.setImageProvider('LocalImageLoader')

    # store all raw and label images of the frames we need for merger resolving:
    labelImages = {}
    rawImages = {}

    # ------------------------------------------------------------
    
    unresolvedGraph = createUnresolvedGraph(divisionsPerTimestep, mergersPerTimestep, mergerLinks)
    resolvedGraph = prepareResolvedGraph(unresolvedGraph)
    refineMergerSegmentations(resolvedGraph, 
                                unresolvedGraph, 
                                timesteps, 
                                pluginManager, 
                                args.label_image_filename, 
                                args.label_image_path)

    # ------------------------------------------------------------
    # compute new object features and edge costs!
    print("Loading raw data")
    for t in labelImages.keys():
        rawImages[t] = pluginManager.getImageProvider().getImageDataAtTimeFrame(
            args.raw_filename, args.raw_path, int(t))

    print("Computing object features")
    objectFeatures = {}
    imageShape = pluginManager.getImageProvider().getImageShape(args.label_image_filename, args.label_image_path)
    print("Found image of shape", imageShape)
    ndims = len(np.array(imageShape).squeeze()) - 1 # get rid of axes with length 1, and minus time axis
    print("Data has dimensionality ", ndims)
    for node in resolvedGraph.nodes_iter():
        intT, idx = node
        # mask out this object only and compute features
        mask = labelImages[str(intT)].copy()
        mask[mask != idx] = 0
        mask[mask == idx] = 1
        
        # compute features, transform to one dict for frame
        frameFeatureDicts, ignoreNames = pluginManager.applyObjectFeatureComputationPlugins(
            ndims, rawImages[str(intT)], labelImages[str(intT)], intT, args.raw_filename)
        frameFeatureItems = []
        for f in frameFeatureDicts:
            frameFeatureItems = frameFeatureItems + f.items()
        frameFeatures = dict(frameFeatureItems)

        # extract all features for this one object
        objectFeatureDict = {}
        for k, v in frameFeatures.iteritems():
            if k in ignoreNames:
                continue
            elif 'Polygon' in k:
                objectFeatureDict[k] = v[1]
            else:
                objectFeatureDict[k] = v[1, ...]
        objectFeatures[node] = objectFeatureDict

    print("Updating edge energy")
    if args.transition_classifier_filename is not None:
        print("\tLoading transition classifier")
        transitionClassifier = traxelstore.RandomForestClassifier(
            args.transition_classifier_path, args.transition_classifier_filename)
    else:
        print("\tUsing distance based transition energies")
        transitionClassifier = None


    # ------------------------------------------------------------
    # run min-cost max-flow to find merger assignments
    print("Running min-cost max-flow to find resolved merger assignments")

    # transform to our graph format:
    segmentationHypotheses = []
    nextId = 0
    
    for node in resolvedGraph.nodes_iter():
        o = {}
        o['id'] = nextId
        resolvedGraph.node[node]['id'] = nextId
        nextId += 1
        o['features'] = [[0], [1]]
        if len(resolvedGraph.in_edges(node)) == 0:
            o['appearanceFeatures'] = [[0], [0]]
        if len(resolvedGraph.out_edges(node)) == 0:
            o['disappearanceFeatures'] = [[0], [0]]
        segmentationHypotheses.append(o)

    def negLog(features):
        fa = np.array(features)
        fa[fa < 0.0000000001] = 0.0000000001
        return list(np.log(fa) * -1.0)

    def listify(l):
        return [[e] for e in l]

    linkingHypotheses = []
    for edge in resolvedGraph.edges_iter():
        e = {}
        e['src'] = resolvedGraph.node[edge[0]]['id']
        e['dest'] = resolvedGraph.node[edge[1]]['id']

        featuresAtSrc = objectFeatures[edge[0]]
        featuresAtDest = objectFeatures[edge[1]]

        if transitionClassifier is not None:
            featVec = pluginManager.applyTransitionFeatureVectorConstructionPlugins(
                featuresAtSrc, featuresAtDest, transitionClassifier.selectedFeatures)
            featVec = np.expand_dims(np.array(featVec), axis=0)
            probs = transitionClassifier.predictProbabilities(featVec)[0]
        else:
            dist = np.linalg.norm(featuresAtDest['RegionCenter'] - featuresAtSrc['RegionCenter'])
            prob = np.exp(-dist / args.trans_par)
            probs = [1.0 - prob, prob]

        e['features'] = listify(negLog(probs))
        linkingHypotheses.append(e)

    trackingGraph = {}
    trackingGraph['segmentationHypotheses'] = segmentationHypotheses
    trackingGraph['linkingHypotheses'] = linkingHypotheses
    trackingGraph['exclusions'] = []

    settings = {}
    settings['statesShareWeights'] = True
    trackingGraph['settings'] = settings

    # track
    import dpct
    weights = {"weights": [1, 1, 1, 1]}
    mergerResult = dpct.trackMaxFlow(trackingGraph, weights)

    # transform results to dictionaries that can be indexed by id or (src,dest)
    nodeFlowMap = dict([(int(d['id']), int(d['value'])) for d in mergerResult['detectionResults']])
    arcFlowMap = dict([((int(l['src']), int(l['dest'])), int(l['value'])) for l in mergerResult['linkingResults']])

    # ------------------------------------------------------------
    # fuse results into a new solution

    # 1.) replace merger nodes in JSON graph by their replacements -> new JSON graph
    #     update UUID to traxel map.
    #     a) how do we deal with the smaller number of states? 
    #        Does it matter as we're done with tracking anyway..?

    # remove merger detections
    def mergerNodeFilter(jsonNode):
        uuid = int(jsonNode['id'])
        traxels = uuidToTraxelMap[uuid]
        return not any(t[1] in mergersPerTimestep[str(t[0])] for t in traxels)

    model['segmentationHypotheses'] = filter(mergerNodeFilter, model['segmentationHypotheses'])

    # remove merger links
    def mergerLinkFilter(jsonLink):
        srcUuid = int(jsonLink['src'])
        destUuid = int(jsonLink['dest'])
        srcTraxels = uuidToTraxelMap[srcUuid]
        destTraxels = uuidToTraxelMap[destUuid]
        return not any((str(destT[0]), (srcT[1], destT[1])) in mergerLinks for srcT, destT in itertools.product(srcTraxels, destTraxels))

    model['linkingHypotheses'] = filter(mergerLinkFilter, model['linkingHypotheses'])

    # insert new nodes and update UUID to traxel map
    nextUuid = max(uuidToTraxelMap.keys()) + 1
    for node in unresolvedGraph.nodes_iter():
        if unresolvedGraph.node[node]['count'] > 1:
            newIds = unresolvedGraph.node[node]['newIds']
            del traxelIdPerTimestepToUniqueIdMap[str(node[0])][str(node[1])]
            for newId in newIds:
                newDetection = {}
                newDetection['id'] = nextUuid
                newDetection['timestep'] = [node[0], node[0]]
                model['segmentationHypotheses'].append(newDetection)
                traxelIdPerTimestepToUniqueIdMap[str(node[0])][str(newId)] = nextUuid
                nextUuid += 1

    # insert new links
    for edge in resolvedGraph.edges_iter():
        newLink = {}
        newLink['src'] = traxelIdPerTimestepToUniqueIdMap[str(edge[0][0])][str(edge[0][1])]
        newLink['dest'] = traxelIdPerTimestepToUniqueIdMap[str(edge[1][0])][str(edge[1][1])]
        model['linkingHypotheses'].append(newLink)

    # save
    core.jsongraph.writeToFormattedJSON(args.out_model_filename, model)
    
    # 
    # 2.) new result = union(old result, resolved mergers) - old mergers

    # filter merger edges
    result['detectionResults'] = filter(mergerNodeFilter, result['detectionResults'])
    result['linkingResults'] = filter(mergerLinkFilter, result['linkingResults'])

    # add new nodes
    for node in unresolvedGraph.nodes_iter():
        if unresolvedGraph.node[node]['count'] > 1:
            newIds = unresolvedGraph.node[node]['newIds']
            for newId in newIds:
                uuid = traxelIdPerTimestepToUniqueIdMap[str(node[0])][str(newId)]
                resolvedNode = (node[0], newId)
                resolvedResultId = resolvedGraph.node[resolvedNode]['id']
                newDetection = { 'id': uuid, 'value': nodeFlowMap[resolvedResultId] }
                result['detectionResults'].append(newDetection)

    # add new links
    for edge in resolvedGraph.edges_iter():
        newLink = {}
        newLink['src'] = traxelIdPerTimestepToUniqueIdMap[str(edge[0][0])][str(edge[0][1])]
        newLink['dest'] = traxelIdPerTimestepToUniqueIdMap[str(edge[1][0])][str(edge[1][1])]
        srcId = resolvedGraph.node[edge[0]]['id']
        destId = resolvedGraph.node[edge[1]]['id']
        newLink['value'] = arcFlowMap[(srcId, destId)]
        result['linkingResults'].append(newLink)

    # save
    core.jsongraph.writeToFormattedJSON(args.out_result, result)

    # 
    # 3.) export refined segmentation
    h5py.File(args.out_label_image, 'w').close()
    for t in timesteps:
        pluginManager.getImageProvider().exportLabelImage(labelImages[t], int(t), args.out_label_image, args.label_image_path)
