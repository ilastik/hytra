# pythonpath modification to make hytra and empryonic available 
# for import without requiring it to be installed
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
# standard imports
try:
    import commentjson as json
except ImportError:
    import json
import argparse
import numpy as np
import copy
from hytra.util.progressbar import ProgressBar

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Replicate nodes, links, divisions and exclusion sets N times, ' \
        'so that the total number of timeframes does not change',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', required=True, type=str, dest='model_filename',
                        help='Filename of the json model description')
    parser.add_argument('--output', required=True, type=str, dest='result_filename',
                        help='Filename of the json file that will hold the replicated model')
    parser.add_argument('--num', type=int, dest='num', default=2,
                        help='how many instances of the original model shall be present in the result file')
    
    args = parser.parse_args()

    print("Loading model file: " + args.model_filename)
    with open(args.model_filename, 'r') as f:
        model = json.load(f)

    segmentationHypotheses = model['segmentationHypotheses']
    # use generator expression instead of list comprehension, we only need it once!
    maxId = max((i['id'] for i in segmentationHypotheses)) 
    newModel = copy.deepcopy(model)

    for i in range(1, args.num):
        offset = i * (maxId + 1000000) # create random gap in IDs

        for seg in segmentationHypotheses:
            newSeg = copy.deepcopy(seg)
            newSeg['id'] = offset + newSeg['id']
            newSeg['disappTarget'] = i # specify that this should use a different target
            newModel['segmentationHypotheses'].append(newSeg)


        linkingHypotheses = model['linkingHypotheses']
        for link in linkingHypotheses:
            newLink = copy.deepcopy(link)
            newLink['src'] = offset + newLink['src']
            newLink['dest'] = offset + newLink['dest']
            newModel['linkingHypotheses'].append(newLink)

        if 'exclusions' in model:
            for e in model['exclusions']:
                newExclusion = [x + offset for x in e]
                newModel['exclusions'].append(newExclusion)

        if 'divisions' in model:
            for d in model['divisions']:
                newDiv = copy.deepcopy(d)
                newDiv['parent'] = offset + d['parent']
                newDiv['children'] = [offset + c for c in d['children']]
                newModel['divisions'].append(newDiv)
            
    with open(args.result_filename, 'w') as f:
        json.dump(newModel, f, indent=4, separators=(',', ': '))


# python replicate_graph.py --model /Users/chaubold/GoogleDrive/Jobs/IWRHeidelberg/eccv16data/rapoport/graphDistTransitionsConvex.json --output /Users/chaubold/GoogleDrive/Jobs/IWRHeidelberg/eccv16data/rapoport/graphDistTransitionsConvex-2times.json --num 2
# python replicate_graph.py --model /Users/chaubold/GoogleDrive/Jobs/IWRHeidelberg/eccv16data/drosophila/graphNoTransClassConvex.json --output /Users/chaubold/GoogleDrive/Jobs/IWRHeidelberg/eccv16data/drosophila/graphNoTransClassConvex-2times.json --num 2