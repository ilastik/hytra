'''
Utilities that help with loading / saving as well as constructing and parsing
hypotheses graphs stored in our json (or python dictionary) format.
'''

import commentjson as json
import logging
import numpy as np

def writeToFormattedJSON(filename, dictionary):
    '''
    Write a dictionary to JSON, but use proper readable formatting
    '''
    with open(filename, 'w') as f:
        json.dump(dictionary, f, indent=4, separators=(',', ': '))

def getMappingsBetweenUUIDsAndTraxels(model):
    '''
    From a dictionary encoded model, load the "traxelToUniqueId" mapping,
    create a reverse mapping, and return both.
    '''

    # create reverse mapping from json uuid to (timestep,ID)
    traxelIdPerTimestepToUniqueIdMap = model['traxelToUniqueId']
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

def getMergersDetectionsLinksDivisions(result, uuidToTraxelMap, withDivisions):
    # load results and map indices
    mergers = [timestepIdTuple + (entry['value'],) for entry in result['detectionResults'] if entry['value'] > 1 for timestepIdTuple in uuidToTraxelMap[int(entry['id'])]]
    detections = [timestepIdTuple for entry in result['detectionResults'] if entry['value'] > 0 for timestepIdTuple in uuidToTraxelMap[int(entry['id'])]]
    if withDivisions:
        divisions = [uuidToTraxelMap[int(entry['id'])][-1] for entry in result['divisionResults'] if entry['value'] == True]
    else:
        divisions = None
    links = [(uuidToTraxelMap[int(entry['src'])][-1], uuidToTraxelMap[int(entry['dest'])][0]) for entry in result['linkingResults'] if entry['value'] > 0]

    # add all internal links of tracklets
    for v in uuidToTraxelMap.values():
        prev = None
        for timestepIdTuple in v:
            if prev is not None:
                links.append((prev, timestepIdTuple))
            prev = timestepIdTuple

    return mergers, detections, links, divisions

def getMergersPerTimestep(mergers, timesteps):
    ''' returns mergersPerTimestep = { "<timestep>": {<idx>: <count>, <idx>: <count>, ...}, "<timestep>": {...}, ... } '''
    mergersPerTimestep = dict([(t, dict([(idx, count) for timestep, idx, count in mergers if timestep == int(t)])) for t in timesteps])
    return mergersPerTimestep

def getDetectionsPerTimestep(detections, timesteps):
    ''' returns detectionsPerTimestep = { "<timestep>": [<idx>, <idx>, ...], "<timestep>": [...], ...} '''
    detectionsPerTimestep = dict([(t, [idx for timestep, idx in detections if timestep == int(t)]) for t in timesteps])
    return detectionsPerTimestep
    
def getLinksPerTimestep(links, timesteps):
    ''' returns linksPerTimestep = { "<timestep>": [(<idxA> (at previous timestep), <idxB> (at timestep)), (<idxA>, <idxB>), ...], ...} '''
    linksPerTimestep = dict([(t, [(a[1], b[1]) for a, b in links if b[0] == int(t)]) for t in timesteps])
    return linksPerTimestep

def getMergerLinks(linksPerTimestep, mergersPerTimestep, timesteps):
    """ returns merger links as triplets [("timestep", (sourceIdAtTMinus1, destIdAtT)), (), ...]"""
    # filter links: at least one of the two incident nodes must be a merger 
    # for it to be added to the merger resolving graph
    mergerLinks = [(t,(a, b)) for t in timesteps for a, b in linksPerTimestep[t] if a in mergersPerTimestep[str(int(t)-1)] or b in mergersPerTimestep[t]]
    return mergerLinks

def getDivisionsPerTimestep(divisions, linksPerTimestep, timesteps, withDivisions):
    ''' returns divisionsPerTimestep = { "<timestep>": {<parentIdx>: [<childIdx>, <childIdx>], ...}, "<timestep>": {...}, ... } '''
    if withDivisions:
        # find children of divisions by looking for the active links
        divisionsPerTimestep = {}
        for t in timesteps:
            divisionsPerTimestep[t] = {}
            for div_timestep, div_idx in divisions:
                if div_timestep == int(t) - 1:
                    # we have an active division of the mother cell "div_idx" in the previous frame
                    children = [b for a,b in linksPerTimestep[t] if a == div_idx]
                    assert(len(children) == 2)
                    divisionsPerTimestep[t][div_idx] = children
    else:
        divisionsPerTimestep = dict([(t,{}) for t in timesteps])

    return divisionsPerTimestep
