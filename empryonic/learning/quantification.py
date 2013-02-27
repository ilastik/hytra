import unittest as _ut
import numpy as np
from empryonic import io as _io
import math

###
### Events as members of sets for the calculation of performance measures
###
class Event( object ):
    @property
    def setid( self ):
        return self._setid

    @property
    def timestep( self ):
        return self._timestep

    def __init__( self, ids, timestep, setid='base' ):
        self._ids = tuple(ids)
        self._timestep = timestep
        self._setid = setid

    def __eq__(self, other):
        '''Attention: operator is not commutative!'''
        if isinstance(other, self.__class__):
            return self.ids == other.ids and self.timestep == other.timestep
        else:
            return False
    def __ne__( self, other):
        return not(self == other)

    def __hash__( self ):
        return hash((self._ids, self._timestep, self.__class__, self._setid))

    @property
    def ids( self ):
        return self._ids

    def translate( self, origin_match, to_match ):
        ''' Create a new equivalent event. '''
        raise NotImplementedError

    def equivalent_to(self, origin_match, to_match, other):
        '''Check, if the 'other' Event corresponds to the given one.

        Other event should be in a different basic set (either 'base' or
        'contestant')
        
        *_match - { events_id: match_to_id  }
        other - Event
        '''
        raise NotImplementedError

    def visible_in_other(self, origin_match, to_match):
        '''Check if Event could be tracked in other basic set, because the participating 
        cells were at least detected.

        *_match - { events_id: match_to_id  }
        '''
        raise NotImplementedError

    ### debugging
    def is_matched(self, origin_match, to_match):
        raise NotImplementedError

class Move( Event ):
    def translate( self, origin_match, to_match, setid ):
        origin_translated = origin_match[self.ids[0]]
        to_translated = to_match[self.ids[1]]
        return Move( (origin_translated, to_translated), self.timestep, setid )
        
    def equivalent_to( self, origin_match, to_match, other):
        origin_translated = origin_match[self.ids[0]]
        to_translated = to_match[self.ids[1]]
        translated_move = Move( (origin_translated, to_translated), self.timestep, other.setid )
        return (translated_move == other)

    def visible_in_other( self, origin_match, to_match):
        return origin_match[self.ids[0]] and to_match[self.ids[1]]

    def is_matched(self,origin_match, to_match):
        if origin_match.has_key(self.ids[0]) and to_match.has_key(self.ids[1]):
            return True
        else:
            return False

    def __repr__( self ):
        return "Move((" + str(self.ids[0]) + ", "+ str(self.ids[1])+ "))"

class Merger( Event ):
    def translate( self, origin_match, to_match, setid ):
        if self.ids[0] != to_match[self.ids[0]]:
            raise RuntimeError('translate: translation not supported')
        
        return Merger( tuple(self.ids), self.timestep, setid )
        
    def equivalent_to( self, origin_match, to_match, other):
        if self.ids[0] != to_match[self.ids[0]]:
            raise RuntimeError('equivalent_to: translation not supported')
        return (self == other)

    def visible_in_other( self, origin_match, to_match):
        return True

    def is_matched(self,origin_match, to_match):
        if to_match.has_key(self.ids[0]):
            return True
        else:
            return False

    def __repr__( self ):
        return "Merger((" + str(self.ids[0]) + ", "+ str(self.ids[1])+ "))"

class Division( Event ):
    def __init__( self, ids, timestep, setid='base' ):
        children = list(ids[1:3])
        children.sort()
        self._ids = (ids[0], children[0], children[1])
        self._timestep = timestep
        self._setid = setid

    def translate( self, origin_match, to_match, setid ):
        origin_translated = origin_match[self.ids[0]]
        to1_translated = to_match[self.ids[1]]
        to2_translated = to_match[self.ids[2]]
        return Division( (origin_translated, to1_translated, to2_translated), self.timestep, setid )

    def equivalent_to( self, origin_match, to_match, other):
        origin_translated = origin_match[self.ids[0]]
        to1_translated = to_match[self.ids[1]]
        to2_translated = to_match[self.ids[2]]
        translated_division = Division( (origin_translated, to1_translated, to2_translated), self.timestep, other.setid )
        return translated_division == other

    def visible_in_other( self, origin_match, to_match):
        return origin_match[self.ids[0]] and to_match[self.ids[1]] and to_match[self.ids[2]]

    def is_matched(self,origin_match, to_match):
        if origin_match.has_key(self.ids[0]) and to_match.has_key(self.ids[1]) and to_match.has_key(self.ids[2]):
            return True
        else:
            return False

    def __repr__( self ):
        return "Division((" + str(self.ids[0]) + ", "+ str(self.ids[1])+", "+ str(self.ids[2])+ "))"


        
class Appearance( Event ):
    def translate( self, origin_match, to_match, setid):
        origin_translated = to_match[self.ids[0]]
        return Appearance( (origin_translated,), self.timestep, setid )

    def equivalent_to( self, origin_match, to_match, other):
        origin_translated = to_match[self.ids[0]]
        translated_appearance = Appearance( (origin_translated,), self.timestep, other.setid )
        return translated_appearance == other

    def visible_in_other( self, origin_match, to_match):
        return to_match[self.ids[0]]

    def is_matched(self,origin_match, to_match):
        if to_match.has_key(self.ids[0]):
            return True
        else:
            return False

    def __repr__( self ):
       return "Appearance((" + str(self.ids[0]) + ",))"


class Disappearance( Event ):
    def translate( self, origin_match, to_match, setid):
        origin_translated = origin_match[self.ids[0]]
        return Disappearance( (origin_translated,), self.timestep, setid )

    def equivalent_to( self, origin_match, to_match, other):
        origin_translated = origin_match[self.ids[0]]
        translated_disappearance = Disappearance( (origin_translated,), self.timestep, other.setid )
        return translated_disappearance == other

    def visible_in_other( self, origin_match, to_match):
        return origin_match[self.ids[0]]
    
    def is_matched(self,origin_match, to_match):
        if origin_match.has_key(self.ids[0]):
            return True
        else:
            return False

    def __repr__( self ):
        return "Disappearance((" + str(self.ids[0]) + ",))"

###
### routines to create and work with Event Sets
###
def event_set_from( lineageH5, setid='base' ):
    '''Extract tracking results from a lineageH5 file as Events.

    Tracking information has to be present, of course.

    Returns a Set of Event objects.
    ''' 
    events = set()

    mov_ids = lineageH5.get_moves()
    for mov in mov_ids:
        e = Move((mov[0], mov[1]), lineageH5.timestep, setid)
        events.add(e)

    merg_ids = lineageH5.get_mergers()
    for merg in merg_ids:
        e = Merger((merg[0], merg[1]), lineageH5.timestep, setid)
        events.add(e)
    
    div_ids = lineageH5.get_divisions()
    for div in div_ids:
        e = Division((div[0], div[1], div[2]), lineageH5.timestep, setid)
        events.add(e)

    app_ids = lineageH5.get_appearances()
    for app in app_ids:
        if isinstance(app, np.ndarray):
            app = app[0]
        e = Appearance((app,), lineageH5.timestep, setid)
        events.add(e)

    dis_ids = lineageH5.get_disappearances()
    for dis in dis_ids:
        if isinstance(dis, np.ndarray):
            dis = dis[0]
        e = Disappearance((dis,), lineageH5.timestep, setid)
        events.add(e)
    return events



def subset_by_correspondence(match_prev, match_curr, events, other_events):
    '''Divide set into matched and unmatched events.

    events - set to be divided
    other_events - set to compare to
    match_* - { events_id: match_to_id  }

    Return subset of matched events.
    '''
    ret = set()
    if not other_events:
        return ret
    for oe in other_events:
        setid = oe.setid
        break
    def reducer( e ):
        translated = e.translate(match_prev, match_curr, setid)
        if translated and translated in other_events:
            ret.add(e)

    map(reducer, events)
    #for e in events:
    #    translated = e.translate(match_prev, match_curr)
    #    if translated and translated in list(other_events): # sets compare by hash, lists by eq
    #        ret.add(e)
    return ret




def subset_by_visibility(match_prev, match_curr, events):
    '''Divide set by matching detections in the other set.

    Correspondence by visibility is a precondition for full
    correspondence.

    events - set to be divided
    match_* - { events_id: match_to_id  }
    
    Return the subset of events, that are potentially detectable in the other set.
    '''
    ret = set()
    for e in events:
        if e.visible_in_other(match_prev, match_curr):
                ret.add(e)
    return ret    



def by_type( events, type=Event ):
    '''Filter iterable events by type (Move, Division etc.) and return a list
    of passed Events.'''
    s = set([ e for e in events if isinstance(e, type) ])
    return s


###
### Classification of Events and performance measures
###
class Taxonomy( object ):
    '''Classification of elements in base and contestant event sets.'''

    def __init__( self, s ):
        '''The following sets are expected to be present as keys in the 's' dictionary::
        - base_basic # all Events in base set
        - cont_basic # all Events in contestant set

        - base_v # Events are visible in contestant set
        - base_c # Events have corresponding Events in contestant set
        - cont_v
        - cont_c

        - base_d # deficient due to tracking or detection error
        - base_dd # deficient due to detection error
        - base_dt # deficient due to tracking error
        - cont_d
        - cont_dd
        - cont_dt
        '''
        self.base_basic = s["base_basic"]
        self.cont_basic = s["cont_basic"]
        self.base_v = s["base_v"]
        self.base_c = s["base_c"]
        self.cont_v = s["cont_v"]
        self.cont_c = s["cont_c"]
        self.base_d = s["base_d"]
        self.base_dd = s["base_dd"]
        self.base_dt = s["base_dt"]
        self.cont_d = s["cont_d"]
        self.cont_dd = s["cont_dd"]
        self.cont_dt = s["cont_dt"]

        if not self._is_valid():
            raise Exception("Taxonomy::__init__: (sub)-sets don't fit together")

    def _is_valid( self ):
        '''Test if (sub)-sets fit together.'''
        b = []
        b.append(len(self.base_c) == len(self.cont_c))
        b.append(self.base_c.issubset(self.base_v))
        b.append(self.cont_c.issubset(self.cont_v))
        b.append(self.cont_d == self.cont_dd.union(self.cont_dt))
        b.append(self.base_d == self.base_dd.union(self.base_dt))
        if not reduce(lambda r, v: r == v, b):
            return False
        else:
            return True

    def union( self, other ):
        '''Return a union between this and another Taxonomy.'''
        if not other._is_valid():
            raise Exception("Taxonomy::union(): other is not a valid Taxonomy)")
        s = {
            'base_basic': self.base_basic.union( other.base_basic ),
            'cont_basic':  self.cont_basic.union( other.cont_basic ),
            'base_v':  self.base_v.union( other.base_v ),
            'base_c':  self.base_c.union( other.base_c ),
            'cont_v':  self.cont_v.union( other.cont_v ),
            'cont_c':  self.cont_c.union( other.cont_c ),
            'base_d':  self.base_d.union( other.base_d ),
            'base_dd':  self.base_dd.union( other.base_dd ),
            'base_dt':  self.base_dt.union( other.base_dt ),
            'cont_d':  self.cont_d.union( other.cont_d ),
            'cont_dd':  self.cont_dd.union( other.cont_dd ),
            'cont_dt':  self.cont_dt.union( other.cont_dt )
            }
        return Taxonomy( s )

    @staticmethod
    def _safe_frac(enum, denom):
        if enum == 0 and denom == 0:
            return float('nan')
        else:
            return 1.0 * enum/denom

    def precision(self, type=Event):
        enum = len(by_type(self.cont_c, type))
        denom = len(by_type(self.cont_basic, type))
        return self._safe_frac(enum, denom)
        
    def precision_given_visibility(self, type=Event):
        enum = len(by_type(self.cont_c, type))
        denom = len(by_type(self.cont_basic, type)) - len(by_type(self.cont_dd, type))
        return self._safe_frac(enum, denom)
        
    def recall(self, type=Event):
        enum = len(by_type(self.base_c, type))
        denom = len(by_type(self.base_basic, type))
        return self._safe_frac(enum, denom)

    def recall_given_visibility(self, type=Event):
        enum = len(by_type(self.base_c, type))
        denom = len(by_type(self.base_basic, type)) - len(by_type(self.base_dd, type))
        return self._safe_frac(enum, denom)

    def f_measure(self, type=Event):
        enum = len(by_type(self.base_c, type).union(by_type(self.cont_c, type)))
        a = (len(by_type(self.base_c, type)) + len(by_type(self.cont_c, type)))
        assert(enum == a ), str(enum) + " " + str(a)
        denom = len(by_type(self.base_basic, type).union(by_type(self.cont_basic, type)))
        f = self._safe_frac(enum, denom)
        #if not math.isnan(f):
        #    f_check = 2 * self.precision(type) * self.recall(type) / (self.precision(type) + self.recall(type))
        #    if not math.isnan(f_check):
        #        assert( abs(f - f_check) < 0.0000001), str(f) + " - " + str(f_check)
        return f

    def f_measure_given_visibility(self, type=Event):
        enum = len(by_type(self.base_c, type).union(by_type(self.cont_c, type)))
        denom = len(by_type(self.base_basic, type).union(by_type(self.cont_basic, type))) \
            - len(by_type(self.base_dd, type)) \
            - len(by_type(self.cont_dd, type)) 
        return self._safe_frac(enum, denom)

    def all_stats( self ):
        """Fill a dictionary with all kinds of stats."""
        return {
            "n_base": len(self.base_basic),
            "n_cont": len(self.cont_basic),
            "precision": self.precision(),
            "recall": self.recall(),
            "f_measure": self.f_measure(),

            "mov_n_base": len(by_type(self.base_basic, Move)),
            "mov_n_cont": len(by_type(self.cont_basic, Move)),            
            "mov_prec":  self.precision(Move),
            "mov_rec": self.recall(Move),
            "mov_f": self.f_measure(Move),

            "merg_n_base": len(by_type(self.base_basic, Merger)),
            "merg_n_cont": len(by_type(self.cont_basic, Merger)),            
            "merg_prec":  self.precision(Merger),
            "merg_rec": self.recall(Merger),
            "merg_f": self.f_measure(Merger),

            "div_n_base": len(by_type(self.base_basic, Division)),
            "div_n_cont": len(by_type(self.cont_basic, Division)),                        
            "div_prec": self.precision(Division),
            "div_rec": self.recall(Division),
            "div_f":  self.f_measure(Division),

            "app_n_base": len(by_type(self.base_basic, Appearance)),
            "app_n_cont": len(by_type(self.cont_basic, Appearance)),            
            "app_prec": self.precision(Appearance),
            "app_rec": self.recall(Appearance),
            "app_f":  self.f_measure(Appearance),

            "dis_n_base": len(by_type(self.base_basic, Disappearance)),
            "dis_n_cont": len(by_type(self.cont_basic, Disappearance)),            
            "dis_prec": self.precision(Disappearance),
            "dis_rec": self.recall(Disappearance),
            "dis_f": self.f_measure(Disappearance),

            "n_base_v": len(self.base_v),
            "n_cont_v": len(self.cont_v),
            "precision_v": self.precision_given_visibility(),
            "recall_v": self.recall_given_visibility(),
            "f_measure_v": self.f_measure_given_visibility(),

            "mov_n_base_v": len(by_type(self.base_v, Move)),
            "mov_n_cont_v": len(by_type(self.cont_v, Move)),            
            "mov_prec_v":  self.precision_given_visibility(Move),
            "mov_rec_v": self.recall_given_visibility(Move),
            "mov_f_v": self.f_measure_given_visibility(Move),

            "merg_n_base_v": len(by_type(self.base_basic, Merger)),
            "merg_n_cont_v": len(by_type(self.cont_basic, Merger)),            
            "merg_prec_v":  self.precision(Merger),
            "merg_rec_v": self.recall(Merger),
            "merg_f_v": self.f_measure(Merger),

            "div_n_base_v": len(by_type(self.base_v, Division)),
            "div_n_cont_v": len(by_type(self.cont_v, Division)),                        
            "div_prec_v": self.precision_given_visibility(Division),
            "div_rec_v": self.recall_given_visibility(Division),
            "div_f_v":  self.f_measure_given_visibility(Division),

            "app_n_base_v": len(by_type(self.base_v, Appearance)),
            "app_n_cont_v": len(by_type(self.cont_v, Appearance)),            
            "app_prec_v": self.precision_given_visibility(Appearance),
            "app_rec_v": self.recall_given_visibility(Appearance),
            "app_f_v":  self.f_measure_given_visibility(Appearance),

            "dis_n_base_v": len(by_type(self.base_v, Disappearance)),
            "dis_n_cont_v": len(by_type(self.cont_v, Disappearance)),            
            "dis_prec_v": self.precision_given_visibility(Disappearance),
            "dis_rec_v": self.recall_given_visibility(Disappearance),
            "dis_f_v": self.f_measure_given_visibility(Disappearance)
            }
    
    def to_line( self ):
        line = '%(n_base)d,%(n_cont)d,%(precision).4f,%(recall).4f,%(f_measure).4f,%(mov_n_base)d,%(mov_n_cont)d,%(mov_prec).4f,%(mov_rec).4f,%(mov_f).4f,%(merg_n_base)d,%(merg_n_cont)d,%(merg_prec).4f,%(merg_rec).4f,%(merg_f).4f,%(div_n_base)d,%(div_n_cont)d,%(div_prec).4f,%(div_rec).4f,%(div_f).4f,%(app_n_base)d,%(app_n_cont)d,%(app_prec).4f,%(app_rec).4f,%(app_f).4f,%(dis_n_base)d,%(dis_n_cont)d,%(dis_prec).4f,%(dis_rec).4f' % self.all_stats()
        return line

    def __str__( self ):
        pretty = \
'''[overall]
n_base = %(n_base)d
n_contestant = %(n_cont)d
precision = %(precision).4f
recall = %(recall).4f
f_measure = %(f_measure).4f:

[move]
n_base = %(mov_n_base)d
n_contestant = %(mov_n_cont)d
precision = %(mov_prec).4f
recall = %(mov_rec).4f
f_measure = %(mov_f).4f

[merger]
n_base = %(merg_n_base)d
n_contestant = %(merg_n_cont)d
precision = %(merg_prec).4f
recall = %(merg_rec).4f
f_measure = %(merg_f).4f

[division]
n_base = %(div_n_base)d
n_contestant = %(div_n_cont)d
precision = %(div_prec).4f
recall = %(div_rec).4f
f_measure = %(div_f).4f

[appearance]
n_base = %(app_n_base)d
n_contestant = %(app_n_cont)d
precision = %(app_prec).4f
recall = %(app_rec).4f
f_measure = %(app_f).4f

[disappearance]
n_base = %(dis_n_base)d
n_contestant = %(dis_n_cont)d
precision = %(dis_prec).4f
recall = %(dis_rec).4f
f_measure = %(dis_f).4f

[overall|visibility]
n_base = %(n_base_v)d
n_contestant = %(n_cont_v)d
precision = %(precision_v).4f
recall = %(recall_v).4f
f_measure = %(f_measure_v).4f

[move|visibility]
n_base = %(mov_n_base_v)d
n_contestant = %(mov_n_cont_v)d
precision = %(mov_prec_v).4f
recall = %(mov_rec_v).4f
f_measure = %(mov_f_v).4f

[division|visibility]
n_base = %(div_n_base_v)d
n_contestant = %(div_n_cont_v)d
precision = %(div_prec_v).4f
recall = %(div_rec_v).4f
f_measure = %(div_f_v).4f

[appearance|visibility]
n_base = %(app_n_base_v)d
n_contestant = %(app_n_cont_v)d
precision = %(app_prec_v).4f
recall = %(app_rec_v).4f
f_measure = %(app_f_v).4f

[disappearance|visibility]
n_base = %(dis_n_base_v)d
n_contestant = %(dis_n_cont_v)d
precision = %(dis_prec_v).4f
recall = %(dis_rec_v).4f
f_measure = %(dis_f_v).4f
''' % self.all_stats()
        return pretty


def classify_event_sets(base_events, cont_events, prev_assoc, curr_assoc):    
    '''Return a Taxonomy of the elements in base and contestant set.

    *_events - Set of Event
    *_assoc - {'lhs': { base_id: cont_id}, 'rhs': { cont_id: base_id}}
    
    (in general, 'lhs' is considered as base and 'rhs' is contestant)
    '''
    t = dict()
    t["base_basic"] = base_events
    t["cont_basic"] = cont_events

    # 'positive' sets, that contain (partially) matched events
    t["base_v"] = subset_by_visibility(prev_assoc['lhs'], curr_assoc['lhs'], t["base_basic"])
    t["base_c"] = subset_by_correspondence(prev_assoc['lhs'], curr_assoc['lhs'], t["base_basic"], t["cont_basic"])

    t["cont_v"] = subset_by_visibility(prev_assoc['rhs'], curr_assoc['rhs'], t["cont_basic"])
    t["cont_c"] = subset_by_correspondence(prev_assoc['rhs'], curr_assoc['rhs'], t["cont_basic"], t["base_basic"])
    assert(len(t["base_c"]) == len(t["cont_c"])), str(len(t["base_c"])) + " " +str(len(t["cont_c"]))

    # intersections from the 'positive' sets, that contain spurious events
    # dd: deficient due to detection error
    # dt: deficient due to tracking error (given correct detection)
    # d: deficient due to tracking or detection error
    t["base_d"] = t["base_basic"].difference(t["base_c"])
    t["base_dd"] = t["base_basic"].difference(t["base_v"])
    t["base_dt"] = t["base_v"].difference(t["base_c"])
    t["cont_d"] = t["cont_basic"].difference(t["cont_c"])
    t["cont_dd"] = t["cont_basic"].difference(t["cont_v"])
    t["cont_dt"] = t["cont_v"].difference(t["cont_c"])
    
    return Taxonomy(t)



def compute_taxonomy(prev_assoc, curr_assoc, base_fn, cont_fn, timestep=0):
    ''' Convenience function around 'classify_event_sets'.

    *_fn - LineageH5 filename
    '''
    with _io.LineageH5(base_fn, 'r', timestep=timestep) as f:
        base_events = event_set_from( f, 'base' )
    del f
    with _io.LineageH5(cont_fn, 'r', timestep=timestep) as f:
        cont_events = event_set_from( f, 'cont' )
    del f

    # check, if events and assocs fit together
    for e in base_events:
        if not e.is_matched(prev_assoc['lhs'], curr_assoc['lhs']):
            raise Exception("Base Event %s: id(s) not present in assocs" % str(e))
    for e in cont_events:
        if not e.is_matched(prev_assoc['rhs'], curr_assoc['rhs']):
            raise Exception("Contestant Event %s: id(s) not present in assocs" % str(e))

    return classify_event_sets(base_events, cont_events, prev_assoc, curr_assoc)



###
### Tests
###
class TestDivision( _ut.TestCase ):
    def testEq( self ):
        d1 = Division([11,15,17], 0)
        d2 = Division([11,17,15], 0)
        d3 = Division([15,11,17], 0)
        d4 = Division([15,11,17], 1)
        self.assertTrue(d1 == d1)
        self.assertTrue(d1 == d2)
        self.assertFalse(d1 == d3)
        self.assertFalse(d1 == d4)

class TestTaxonomy( _ut.TestCase ):
    def setUp( self ):
        self.empty = {
            'base_basic': set(),
            'cont_basic': set(),
            'base_v': set(),
            'base_c': set(),
            'cont_v': set(),
            'cont_c': set(),
            'base_d': set(),
            'base_dd': set(),
            'base_dt': set(),
            'cont_d': set(),
            'cont_dd': set(),
            'cont_dt': set(),
            }

        self.typical = {
            'base_basic': set((1,2,3,4,5)),
            'cont_basic': set((6,7,8,9)),
            'base_v': set((1,2,4,5)),
            'base_c': set((1,2)),
            'cont_v': set((6,7,9)),
            'cont_c': set((6,7)),
            'base_d': set((3,4,5)),
            'base_dd': set((3,)),
            'base_dt': set((4,5)),
            'cont_d': set((8,9)),
            'cont_dd': set((8,)),
            'cont_dt': set((9,)),
            }

    def test_constructor( self ):
        # shouldn't throw
        Taxonomy(self.empty)
        Taxonomy(self.typical)

    def test_precision( self ):
        t = Taxonomy(self.typical)
        self.assertEqual(t.precision(type=int), 2./4.)

    def test_precision_given_visibility( self ):
        t = Taxonomy(self.typical)
        self.assertEqual(t.precision_given_visibility(type=int), 2./3.)

    def test_recall( self ):
        t = Taxonomy(self.typical)
        self.assertEqual(t.recall(type=int), 2./5.)

    def test_recall_given_visibility( self ):
        t = Taxonomy(self.typical)
        self.assertEqual(t.recall_given_visibility(type=int), 2./4.)

    def test_f_measure( self ):
        t = Taxonomy(self.typical)
        self.assertEqual(t.f_measure(type=int), 4./9.)

    def test_f_measure_given_visibility( self ):
        t = Taxonomy(self.typical)
        self.assertEqual(t.f_measure_given_visibility(type=int), 4./7.)

class Test_classify_event_sets( _ut.TestCase ):
    def test_typical( self ):
        base_events = set((
                Move((1,1), 0),
                Disappearance((2,), 0),
                Appearance((3,), 0),
                Division((4,5,6), 0),
                Appearance((7,), 0)
                ))
        cont_events = set((
                Move((10,10), 0),
                Move((20,30), 0),
                Division((40,50,60), 0),
                Move((80,90), 0)
                ))
        prev_assoc = {
            'lhs': {
                1: 10,
                2: 20,
                4: 40
                },
            'rhs': {
                10: 1,
                20: 2,
                40: 4,
                80: None}}
        curr_assoc = {
            'lhs': {
                1: 10,
                3: 30,
                5: 50,
                6: 60,
                7: None
                },
            'rhs': {
                10: 1,
                30: 3,
                50: 5,
                60: 6,
                90: None
                }}

        t = classify_event_sets(base_events, cont_events, prev_assoc, curr_assoc)
        import math

        self.assertEqual(t.precision(), 2./4.)
        self.assertEqual(t.recall(), 2./5.)
        self.assertEqual(t.f_measure(), 4./9.)

        self.assertEqual(t.precision(Move), 1./3.)
        self.assertEqual(t.recall(Move), 1.)
        self.assertEqual(t.f_measure(Move), 2./4.)

        self.assertEqual(t.precision(Division), 1.)
        self.assertEqual(t.recall(Division), 1.)
        self.assertEqual(t.f_measure(Division), 2./2.)

        self.assertTrue(math.isnan(t.precision(Appearance)))
        self.assertEqual(t.recall(Appearance), 0.)
        self.assertEqual(t.f_measure(Appearance), 0.)

        self.assertTrue(math.isnan(t.precision(Disappearance)))
        self.assertEqual(t.recall(Disappearance), 0.)
        self.assertEqual(t.f_measure(Disappearance), 0.)

if __name__ == '__main__':
    _ut.main()


