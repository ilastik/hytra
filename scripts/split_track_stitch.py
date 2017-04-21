'''
Given a hypotheses graph and weights, this script tries to find split points where there are not many mergers,
splits the graph into N parts, and tracks them independently.
'''
from __future__ import print_function, absolute_import, nested_scopes, generators, division, with_statement, unicode_literals
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
# standard imports
try:
    import commentjson as json
except ImportError:
    import json
import logging
import copy
import configargparse as argparse
import numpy as np
import networkx as nx
import time
import hytra.core.jsongraph
import concurrent.futures

def _getLogger():
    ''' logger to be used in this module '''
    return logging.getLogger("split-track-stitch")

def track(model, weights, solver='flow'):
    ''' solver may be flow or ilp '''
    if solver == 'flow':
        import dpct
        return dpct.trackFlowBased(model, weights)
    else:
        try:
            import multiHypoTracking_with_cplex as mht
        except ImportError:
            try:
                import multiHypoTracking_with_gurobi as mht
            except ImportError:
                raise ImportError("Could not find multi hypotheses tracking ilp solver")
        return mht.track(model, weights)


def trackAndContractSubmodel(submodel, weights, modelIdx, solver):
    try:
        _getLogger().info("Tracking submodel {}".format(modelIdx))
        result = track(submodel, weights, solver)

        linksByIdTuple = {}
        for l in submodel['linkingHypotheses']:
            linksByIdTuple[(l['src'], l['dest'])] = l

        detectionsById = {}
        for d in submodel['segmentationHypotheses']:
            detectionsById[d['id']] = d

        tracklets = []
        links = []
        nodeIdRemapping = {}
        valuePerDetection = {}
        divisionsPerDetection = {}
            
        # find connected components of graph where edges are only inserted if the value of the nodes agrees with the value along the link
        # at divisions we do not insert the links so that lineage trees are no connected components.
        g = nx.Graph()
        for d in result['detectionResults']:
            valuePerDetection[d['id']] = d['value']
            divisionsPerDetection[d['id']] = False # initialize, will be overwritten below if it is given
            if d['value'] > 0:
                g.add_node(d['id'])

        for d in result['divisionResults']:
            divisionsPerDetection[d['id']] = d['value']

        for l in result['linkingResults']:
            s, d = l['src'], l['dest']
            if l['value'] > 0 and divisionsPerDetection[s] is False and valuePerDetection[s] == l['value'] and valuePerDetection[d] == l['value']:
                g.add_edge(s, d)

        # for every connected component, insert a node into the stitching graph
        connectedComponents = nx.connected_components(g)
        _getLogger().info("Contracting tracks of submodel {}".format(modelIdx))

        for c in connectedComponents:
            # sum over features of dets + links
            linksInTracklet = [idTuple for idTuple in linksByIdTuple.keys() if idTuple[0] in c and idTuple[1] in c]
            linkFeatures = [linksByIdTuple[idTuple]['features'] for idTuple in linksInTracklet]
            detFeatures = [detectionsById[i]['features'] for i in c]
            accumulatedFeatures = np.sum([hytra.core.jsongraph.delistify(f) for f in linkFeatures + detFeatures], axis=0)
            
            # Get tracklet ids from nodes at start and end times of tracklets
            minTime = None
            maxTime = None

            for n in c:
                if maxTime is None or detectionsById[n]['nid'][0] > maxTime:
                    maxTime = detectionsById[n]['nid'][0]
                    maxTrackletId = n
                    
                if minTime is None or detectionsById[n]['nid'][0] < minTime:
                    minTime = detectionsById[n]['nid'][0]
                    minTrackletId = n 
                
            contractedNode = {
                'id' : minTrackletId, 
                'contains' : set(c),
                'links' : linksInTracklet,
                'nid' : detectionsById[minTrackletId]['nid'],
                'minUid' : minTrackletId,
                'maxUid' : maxTrackletId,
                'features' : hytra.core.jsongraph.listify(accumulatedFeatures),
                'timestep' : [minTime, maxTime]
            }

            if 'appearanceFeatures' in detectionsById[minTrackletId]:#min(c)]:
                contractedNode['appearanceFeatures'] = detectionsById[minTrackletId]['appearanceFeatures']
            if 'disappearanceFeatures' in detectionsById[maxTrackletId]:#max(c)
                contractedNode['disappearanceFeatures'] = detectionsById[maxTrackletId]['disappearanceFeatures']
            if 'divisionFeatures' in detectionsById[maxTrackletId]:
                contractedNode['divisionFeatures'] = detectionsById[maxTrackletId]['divisionFeatures']

            tracklets.append(contractedNode)

            for n in c:
                nodeIdRemapping[n] = minTrackletId

        # add the remaining links to the stitching graph with adjusted source and destination
        for l in result['linkingResults']:
            s, d = l['src'], l['dest']
            if l['value'] > 0 and (valuePerDetection[s] != l['value'] or valuePerDetection[d] != l['value'] or divisionsPerDetection[s]):
                newL = {
                    'src' : nodeIdRemapping[s],
                    'dest' : nodeIdRemapping[d],
                    'features' : linksByIdTuple[(s, d)]['features']
                }

                links.append(newL)

        _getLogger().info("Found divisions at {}".format([k for k, v in divisionsPerDetection.iteritems() if v is True]))
        return modelIdx, result, links, tracklets, nodeIdRemapping, valuePerDetection, sum(1 for v in divisionsPerDetection.values() if v is True)
    except:
        _getLogger().exception('Exception while processing submodel')

def main(args):
    assert args.solver in ['flow', 'ilp'], "Invalid Solver selected"

    with open(args.model_filename, 'r') as f:
        model = json.load(f)

    with open(args.weights_filename, 'r') as f:
        weights = json.load(f)
    _getLogger().info("Done loading model and weights")

    traxelIdPerTimestepToUniqueIdMap, uuidToTraxelMap = hytra.core.jsongraph.getMappingsBetweenUUIDsAndTraxels(model)
    assert not any(len(u2t) > 1 for u2t in uuidToTraxelMap.values()), "Doesn't work with tracklets yet!"

    detectionTimestepTuples = [(timestepIdTuple, entry) for entry in model['segmentationHypotheses'] for timestepIdTuple in uuidToTraxelMap[int(entry['id'])]]
    detectionsPerTimestep = {}
    for timestep_id, detection in detectionTimestepTuples:
        detectionsPerTimestep.setdefault(int(timestep_id[0]), []).append(detection)

    nonSingletonCostsPerFrame = []
    detectionsById = {}
    linksByIdTuple = {}

    _getLogger().info("Setup done. Searching for good split locations...")

    for t in detectionsPerTimestep.keys():
        nonSingletonCosts = []
        for d in detectionsPerTimestep[t]:
            detectionsById[d['id']] = d
            d['nid'] = uuidToTraxelMap[d['id']][0]
            f = d['features'][:]
            del f[1]
            nonSingletonCosts.extend(f)
        nonSingletonCostsPerFrame.append(min(nonSingletonCosts)[0])

    for l in model['linkingHypotheses']:
        linksByIdTuple[(l['src'], l['dest'])] = l

    # create a list of the sum of 2 neighboring elements (has len = len(nonSingletonCostsPerFrame) - 1)
    nonSingletonCostsPerFrameGap = [i + j for i, j in zip(nonSingletonCostsPerFrame[:-1], nonSingletonCostsPerFrame[1:])]

    # for debugging: show which frames could be interesting. The higher the value, the more are all objects in the frame true detections.
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(nonSingletonCostsPerFrame)
    # plt.show()

    firstFrame = min(detectionsPerTimestep.keys())
    lastFrame = max(detectionsPerTimestep.keys())
    numSplits = args.num_splits
    numFramesPerSplit = (lastFrame - firstFrame) // numSplits

    # find points where TWO consecutive frames have a low merger score together!
    # find split points in a range of 10 frames before/after the desired split location
    # TODO: also consider divisions!
    splitPoints = []
    border = 10
    if numFramesPerSplit < border*2:
        border = 1

    for s in range(1, numSplits):
        desiredSplitPoint = s * numFramesPerSplit
        subrange = np.array(nonSingletonCostsPerFrameGap[desiredSplitPoint - border : desiredSplitPoint + border])
        splitPoints.append(desiredSplitPoint - border + np.argmax(subrange))

    _getLogger().info("Going to split hypotheses graph at frames {}".format(splitPoints))

    # for debugging: show chosen frames
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(nonSingletonCostsPerFrame)
    # plt.scatter(splitPoints, [nonSingletonCostsPerFrame[s] for s in splitPoints])
    # plt.show()

    # split graph
    def getSubmodel(startTime, endTime):
        # for each split: take detections from detectionsPerTimestep, store a list of the uuids, then add links by filtering for the uuids
        # also make sure that appearance/disappearance costs are zero at the beginning/end of each submodel

        # TODO: tracklets that reach over the gap must be split, or put into just one submodel but connected to the other side?
        submodel = {}
        segmentationHypotheses = []
        for f in range(startTime, endTime):
            if f == startTime:
                for d in detectionsPerTimestep[f]:
                    newD = copy.deepcopy(d)
                    newD['appearanceFeatures'] = [[0.0000001 * sum(range(i+1))] for i in range(len(d['features']))]
                    segmentationHypotheses.append(newD)
            elif f+1 == endTime:
                for d in detectionsPerTimestep[f]:
                    newD = copy.deepcopy(d)
                    newD['disappearanceFeatures'] = [[0.0000001 * sum(range(i+1))] for i in range(len(d['features']))]
                    segmentationHypotheses.append(newD)
            else:
                segmentationHypotheses.extend(detectionsPerTimestep[f])

        submodel['segmentationHypotheses'] = segmentationHypotheses
        uuidsInSubmodel = set([d['id'] for f in range(startTime, endTime) for d in detectionsPerTimestep[f]])
        submodel['linkingHypotheses'] = [l for l in model['linkingHypotheses'] if (l['src'] in uuidsInSubmodel) and (l['dest'] in uuidsInSubmodel)]
        submodel['divisionHypotheses'] = []
        submodel['settings'] = model['settings']
        return submodel

    submodels = []
    lastSplit = 0
    splitPoints.append(lastFrame) # so that we get the last split as well
    for splitPoint in splitPoints:
        _getLogger().info("Creating submodel from t={} to t={}...".format(lastSplit, splitPoint + 1))
        submodels.append(getSubmodel(lastSplit, splitPoint + 1))
        _getLogger().info("\t contains {} nodes and {} edges".format(len(submodels[-1]['segmentationHypotheses']), len(submodels[-1]['linkingHypotheses'])))
        lastSplit = splitPoint + 1

    # We will track in parallel now.
    # how to merge results?
    # make detection weight higher, or accumulate energy over tracks (but what to do with mergers then?),
    # or contract everything where source-node, link and destination have the same number of objects?
    # We choose the last option.
    _getLogger().info("Tracking in parallel and contracting tracks for stitching")
    results = []
    tracklets = []
    links = []
    stitchingModel = {'segmentationHypotheses': tracklets, 'linkingHypotheses': links, 'divisionHypotheses' : [], 'settings' : model['settings']}
    nodeIdRemapping = {}
    valuePerDetection = {}
    numDivisions = 0
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # 1st pass for region features
        jobs = []
        for i, submodel in enumerate(submodels):
            jobs.append(executor.submit(trackAndContractSubmodel,
                                        submodel,
                                        weights,
                                        i,
                                        args.solver
            ))

        for job in concurrent.futures.as_completed(jobs):
            idx, r, l, t, n, v, nd = job.result()
            _getLogger().info("Finished tracking submodel {}".format(idx))
            results.append(r) # can be randomly ordered!
            links.extend(l)
            tracklets.extend(t)
            nodeIdRemapping.update(n)
            valuePerDetection.update(v)
            numDivisions += nd

    _getLogger().info("\tgot {} links from within the submodels".format(len(links)))
    _getLogger().info("\tfound {} divisions within the submodels".format(numDivisions))

    # insert all edges crossing the splits that connect active detections
    detectionIdsPerTimestep = dict( [(k, [d['id'] for d in v]) for k, v in detectionsPerTimestep.iteritems()])
    for idTuple, link in linksByIdTuple.iteritems():
        s, d = idTuple
        for splitPoint in splitPoints[:-1]:
            if s in detectionIdsPerTimestep[splitPoint] and d in detectionIdsPerTimestep[splitPoint + 1] and valuePerDetection[s] > 0 and valuePerDetection[d] > 0:
                newL = copy.deepcopy(link)
                newL['src'] = nodeIdRemapping[s]
                newL['dest'] = nodeIdRemapping[d]
                links.append(newL)

    _getLogger().info("\tcontains {} nodes and {} edges".format(len(tracklets), len(links)))
    # hytra.core.jsongraph.writeToFormattedJSON('/Users/chaubold/Desktop/stitchingGraph.json', stitchingModel)
    stitchingModel['settings']['allowLengthOneTracks'] = True
    stitchingResult = track(stitchingModel, weights, args.solver)
    # hytra.core.jsongraph.writeToFormattedJSON('/Users/chaubold/Desktop/stitchingResult.json', stitchingResult)
    
    _getLogger().info("Extracting stitched result...")

    # extract full result
    trackletsById = dict([(t['id'], t) for t in tracklets])
    fullResult = {'detectionResults' : [], 'linkingResults' : [], 'divisionResults' : []}
    
    t0 = time.time()
    for dr in stitchingResult['detectionResults']:
        v = dr['value'] 
        t = trackletsById[dr['id']]
        if v > 0:
            for originalUuid in t['contains']:
                fullResult['detectionResults'].append({'id': originalUuid, 'value': v})
            for s, d in t['links']:
                fullResult['linkingResults'].append({'src': s, 'dest' : d, 'value': v})
        else:
            _getLogger().debug("Skipped detection {} while stitching!".format(t))

    for lr in stitchingResult['linkingResults']:
        v = lr['value'] 
        st = trackletsById[lr['src']]
        dt = trackletsById[lr['dest']]
        if v > 0:
            fullResult['linkingResults'].append({'src': st['maxUid'], 'dest' : dt['minUid'], 'value': v})

    for dr in stitchingResult['divisionResults']:
        v = dr['value']
        t = trackletsById[dr['id']]
        fullResult['divisionResults'].append({'id': t['maxUid'], 'value': v})
    
    t1 = time.time()
    _getLogger().info("Extracting result took {} secs".format(t1-t0))

    _getLogger().info("Saving stitched result to {}".format(args.results_filename))
    hytra.core.jsongraph.writeToFormattedJSON(args.results_filename, fullResult)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Take a json file containing a result to a set of HDF5 events files',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', is_config_file=True, help='config file path')

    parser.add_argument('--graph-json-file', required=True, type=str, dest='model_filename',
                        help='Filename of the json model description')
    parser.add_argument('--weights-json-file', required=True, type=str, dest='weights_filename',
                        help='Filename of the json file containing weights')
    parser.add_argument('--out-json-file', required=True, type=str, dest='results_filename',
                        help='Filename where to store the results after tracking as JSON')
    parser.add_argument('--num-splits', required=True, type=int, dest='num_splits',
                        help='Into how many pieces the tracking problem should be split')
    parser.add_argument('--solver', default='flow', type=str, dest='solver',
                        help='Solver to use, may be "flow" or "ilp"')
    parser.add_argument("--verbose", dest='verbose', action='store_true', default=False)
    
    args, unknown = parser.parse_known_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    _getLogger().debug("Ignoring unknown parameters: {}".format(unknown))

    args.solver = args.solver.lower()
    
    main(args)










