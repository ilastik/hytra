from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
from builtins import map
import unittest
import numpy as np

from empryonic.tracklets import Tracklet, Tracklets
from empryonic import io as _io
from . import optimal_matching as _om

def idAssoc_from_trackletAssoc( assoc ):
    ''' Construct an id based association dict from a trackled based one.'''
    ret = dict()
    ret['lhs'] = dict()
    ret['rhs'] = dict()
    
    for i,v in list(assoc['lhs'].items()):
        if v:
            ret['lhs'][i.id] = v.id
        else:
            ret['lhs'][i.id] = None
    for i,v in list(assoc['rhs'].items()):
        if v:
            ret['rhs'][i.id] = v.id
        else:
            ret['rhs'][i.id] = None
    return ret


def optimal_matching( lhs, rhs, nonmatch_threshold=30):
    '''Associate two sets of tracklets using globally optimal nearest neighbor matching.

    Returns a dictionary of associations of the form:
    {'lhs': {lhs_instance: rhs_instance}, 'rhs': {rhs_instance: lhs_instance}}

    In case of a nonmatch, the instance is associated with None.

    lhs, rhs -- iterable of unique Tracklet instances 
    nonmatch_threshold -- distance threshold; below are potential matching partners

    '''
    # In case of euclidean distance based nearest neighbor matching, the nonmatch_cost is equivalent
    # with a hard distance cutoff; therefore we set the cutoff to almost the nonmatch_cost. We shouldn't
    # set it exactly to the cost, because there may be problems if some match cost has exactly the value of
    # the nonmatch cost. We better give the optimization scheme some room to breathe.
    match =  _om.optimal_matching( lhs, rhs, Tracklet.distance, lambda x: nonmatch_threshold, nonmatch_threshold + 1)
    return match

    
def greedy_matching( lhs, rhs, cutoff=30):
    '''Match two collections of tracklets using greedy nearest neighbor matching.

    NOTE: function is superseded by optimal_matching.
    '''
    lhs = set(lhs)
    rhs = set(rhs)
    matches = []
    neighborhood = rhs.copy()
    for tracklet in lhs:
        if neighborhood == set():
            break
        dist, nn = tracklet.nearest_neighbor( neighborhood )
        if dist < cutoff:
            matches.append((tracklet, nn))
            neighborhood.remove(nn)

    unmatched_lhs = set.difference(lhs, set([pair[0] for pair in matches]))
    unmatched_rhs = neighborhood

    assoc = {'lhs':{}, 'rhs':{}}

    for match in matches:
        assoc['lhs'][match[0]] = match[1]
        assoc['rhs'][match[1]] = match[0]
    for loner in unmatched_lhs:
        assoc['lhs'][loner] = None
    for loner in unmatched_rhs:
        assoc['rhs'][loner] = None

    return assoc



def match(lhs, rhs, nonmatch_threshold = 25, ignore_z=False, swap_xy=False, method=optimal_matching, verbose=True):
    '''High level matching routine.
    
    Warning: modifies lhs and rhs in place!

    lhs - Tracklets
    rhs - Tracklets
    '''
    # preprocess traxels
    if ignore_z:
        def xy_projector(tr):
            tr.z = 0
        [xy_projector(x) for x in lhs.the]
        [xy_projector(x) for x in rhs.the]
        # list(map(xy_projector, lhs.the))
        # list(map(xy_projector, rhs.the))
        if verbose:
            print("-> Projected traxels to x-y subspace.")
    if swap_xy:    
        lhs = Tracklets(list(lhs.xy_swapped()))
        if verbose:

            print("-> Swapped x and y coordinates.")
    if verbose:
        if verbose:
            print("-- Calling match method")
    assoc = method( lhs, rhs, nonmatch_threshold)
    if verbose:
        print("-> Finished matching.")
    return assoc

def match_files(lhs_h5, rhs_h5, nonmatch_threshold = 25, ignore_z=False, swap_xy=False, method=optimal_matching, verbose=True):
    '''Convenience wrapper around 'match'.
    
    Matches two LineageH5 files.
    *_h5 - filenames
    '''
    with _io.LineageH5(lhs_h5, 'r') as f:
        traxels1 = f.Tracklets(position='max', add_features_as_meta=False)
    del f

    with _io.LineageH5(rhs_h5, 'r') as f:
        traxels2 = f.Tracklets(position='max', add_features_as_meta=False)
    del f
    assoc =  match( traxels1, traxels2, nonmatch_threshold, ignore_z, swap_xy, method, verbose )
    if verbose:
        print("-> matched: " + path.basename(lhs_h5) + " <-> " + path.basename(rhs_h5))
    return idAssoc_from_trackletAssoc( assoc )



class TestOptimalMatching( unittest.TestCase ):
    def runTest( self ):
        lhs = [
            Tracklet(1,10,0,0,0),
            Tracklet(1,5,0,0,0),
            Tracklet(2.5,5.1,0,0,0),
            Tracklet(2.5,5.0,0,0,0),
            Tracklet(2.1,1.9,0,0,0)
            ]

        rhs = [
            Tracklet(1,11,0,0,0),
            Tracklet(1.1,5.2,0,0,0),
            Tracklet(2,2.3,0,0,0),
            Tracklet(2.5,5.0,0,0,0),
            Tracklet(2.2,2.0,0,0,0),
            Tracklet(10,1,0,0,0)
            ]

        expected = {
            'lhs': {
                lhs[0]: None,
                lhs[1]: rhs[1],
                lhs[2]: None,
                lhs[3]: rhs[3],
                lhs[4]: rhs[4],
                },
            'rhs': {
                rhs[0]: None,
                rhs[1]: lhs[1],
                rhs[2]: None,
                rhs[3]: lhs[3],
                rhs[4]: lhs[4],
                rhs[5]: None                
                }
            }
        match = optimal_matching( lhs, rhs, 0.5)
        self.assertEqual(match, expected)
        
if __name__ == '__main__':
    unittest.main()

