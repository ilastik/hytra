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
import hytra.core.jsongraph

def _getLogger():
    ''' logger to be used in this module '''
    return logging.getLogger("split-track-stitch")

class SplitTracking:
    def __init__(self):
        pass
     
    @staticmethod
    def trackFlowBasedWithSplits(model, weights, numSplits):
        logging.basicConfig(level=logging.INFO)

        traxelIdPerTimestepToUniqueIdMap, uuidToTraxelMap = hytra.core.jsongraph.getMappingsBetweenUUIDsAndTraxels(model)
        
        detectionTimestepTuples = [(timestepIdTuple, entry) for entry in model['segmentationHypotheses'] for timestepIdTuple in uuidToTraxelMap[int(entry['id'])]]
        detectionsPerTimestep = {}
        for timestep_id, detection in detectionTimestepTuples:
            detectionsPerTimestep.setdefault(int(timestep_id[0]), []).append(detection)
    
        nonSingletonCostsPerFrame = []
        detectionsById = {}
        linksByIdTuple = {}
    
        for t in detectionsPerTimestep.keys():
            nonSingletonCosts = []
            for d in detectionsPerTimestep[t]:
                detectionsById[d['id']] = d
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

        numFramesPerSplit = (lastFrame - firstFrame) // numSplits
    
        # Check that number of frames per split is more than 2
        assert numFramesPerSplit > 2 , "The number of splits is too large; submodel has less than 2 frames"
    
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
    
            # TODO: tracklets that reach over the gap must be split into two!
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
    
        # run tracking (in parallel??)
        results = []
        import dpct
        for i, submodel in enumerate(submodels):
            # TODO: be robust against changes of num weights!
            # TODO: release GIL in tracking python wrappers to allow parallel solving!!
            _getLogger().info("Tracking submodel {}/{}".format(i, len(submodels)))
            results.append(dpct.trackFlowBased(submodel, weights))
    
        # merge results
        # make detection weight higher, or accumulate energy over tracks (but what to do with mergers then?),
        # or contract everything where source-node, link and destination have the same number of objects?
        # We choose the last option.
        _getLogger().info("Setting up model for stitching")
        tracklets = []
        links = []
        stitchingModel = {'segmentationHypotheses': tracklets, 'linkingHypotheses': links, 'divisionHypotheses' : [], 'settings' : model['settings']}
        nodeIdRemapping = {}
        valuePerDetection = {}
    
        modelIdx = 0
        for submodel, result in zip(submodels, results):
            divisionsPerDetection = {}
            
            # find connected components of graph where edges are only inserted if the value of the nodes agrees with the value along the link
            g = nx.Graph()
            for d in result['detectionResults']:
                valuePerDetection[d['id']] = d['value']
                if 'divisionValue' in d and d['divisionValue']:
                    divisionsPerDetection[d['id']] = True
                else:
                    divisionsPerDetection[d['id']] = False
                g.add_node(d['id'])
    
            for l in result['linkingResults']:
                s, d = l['src'], l['dest']
                if divisionsPerDetection[s] is False and valuePerDetection[s] == l['value'] and valuePerDetection[d] == l['value']:
                    g.add_edge(s, d)
    
            # for every connected component, insert a node into the stitching graph
            connectedComponents = nx.connected_components(g)
            _getLogger().info("Contracting tracks of submodel {}/{}".format(modelIdx, len(submodels)))
    
            for c in connectedComponents:
                # sum over features of dets + links
                linkFeatures = [link['features'] for idTuple, link in linksByIdTuple.iteritems() if idTuple[0] in c and idTuple[1] in c]
                detFeatures = [detectionsById[i]['features'] for i in c]
                accumulatedFeatures = np.sum([hytra.core.jsongraph.delistify(f) for f in linkFeatures + detFeatures], axis=0)
    
                trackletId = min(c)
                contractedNode = {
                    'id' : trackletId, 
                    'contains' : c,
                    'features' : hytra.core.jsongraph.listify(accumulatedFeatures),
                    'appearanceFeatures' : detectionsById[min(c)]['appearanceFeatures'],
                    'disappearanceFeatures' : detectionsById[max(c)]['disappearanceFeatures'],
                }
                if 'divisionFeatures' in detectionsById[max(c)]:
                    contractedNode['divisionFeatures'] = detectionsById[max(c)]['divisionFeatures']
                tracklets.append(contractedNode)
    
                for n in c:
                    nodeIdRemapping[n] = trackletId
    
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
            modelIdx += 1
        _getLogger().info("\tgot {} links from within the submodels".format(len(links)))
    
        # insert all edges crossing the splits that connect active detections
        detectionIdsPerTimestep = dict( [(k, [d['id'] for d in v]) for k, v in detectionsPerTimestep.iteritems()])
        for splitPoint in splitPoints[:-1]:
            for idTuple, link in linksByIdTuple.iteritems():
                s, d = idTuple
                if s in detectionIdsPerTimestep[splitPoint] and d in detectionIdsPerTimestep[splitPoint + 1] and valuePerDetection[s] > 0 and valuePerDetection[d] > 0:
                    newL = copy.deepcopy(link)
                    newL['src'] = nodeIdRemapping[s]
                    newL['dest'] = nodeIdRemapping[d]
                    links.append(newL)
    
        _getLogger().info("\t contains {} nodes and {} edges".format(len(tracklets), len(links)))
        stitchingResult = dpct.trackFlowBased(stitchingModel, weights)
        
        # TODO: extract full result
        trackletsById = dict([(t['id'], t) for t in tracklets])
        fullResult = {'detectionResults' : [], 'linkingResults' : [], 'divisionResults' : []}
        
        for dr in stitchingResult['detectionResults']:
            v = dr['value'] 
            t = trackletsById[dr['id']]
            if v > 0:
                for originalUuid in t['contains']:
                    fullResult['detectionResults'].append({'id': originalUuid, 'value': v})
                for s, d in linksByIdTuple.keys():
                    if s in t['contains'] and d in t['contains']:
                        fullResult['linkingResults'].append({'src': s, 'dest' : d, 'value': v})
            else:
                _getLogger().warning("Skipped detection {} while stitching!".format(t))
    
        for lr in stitchingResult['linkingResults']:
            v = lr['value'] 
            st = trackletsById[lr['src']]
            dt = trackletsById[lr['dest']]
            if v > 0:
                fullResult['linkingResults'].append({'src': max(st['contains']), 'dest' : min(dt['contains']), 'value': v})
    
        return fullResult









