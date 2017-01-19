from __future__ import unicode_literals
from __future__ import division
from past.utils import old_div
from builtins import object
import numpy as np

from empryonic.tracklets import Tracklet

class Move( object ):
    def __init__(self, origin = Tracklet(), to = Tracklet(), energy=None):
        self.origin = origin
        self.to = to
        self.energy = energy

    def distance( self ):
        return self.origin.distance(self.to)

    def vec( self ):
        return (self.to.x - self.origin.x,
                self.to.y - self.origin.y,
                self.to.z - self.origin.z
                )

    def point_of_origin( self ):
        return (self.origin.x, self.origin.y, self.origin.z)
        
    def __repr__( self ):
        return 'Move(%s, %s)' % (repr(self.origin), repr(self.to))



class Division( object ):
    def __init__(self, origin = Tracklet(), to1 = Tracklet(), to2 = Tracklet(), energy=None):
        self.origin = origin
        self.to1 = to1
        self.to2 = to2
        self.energy = energy
    def distance1( self ):
        return self.origin.distance(self.to1)

    def distance2( self ):
        return self.origin.distance(self.to2)

    def point_of_origin( self ):
        return (self.origin.x, self.origin.y, self.origin.z)

    def vec1( self ):
        return (self.to1.x - self.origin.x,
                self.to1.y - self.origin.y,
                self.to1.z - self.origin.z)
    def vec2( self ):
        return (self.to2.x - self.origin.x,
                self.to2.y - self.origin.y,
                self.to2.z - self.origin.z)

    def child_distance( self ):
        return self.to1.distance(self.to2)

    def angle( self ):
        return np.arccos(old_div(np.dot(self.vec1(), self.vec2()),
                         (np.linalg.norm(self.vec1()) * np.linalg.norm(self.vec2()))))

class Appearance( object ):
    def __init__(self, to = Tracklet(), energy=None):
        self.to = to
        self.energy = energy
    def point_of_origin( self ):
        return (self.to.x, self.to.y, self.to.z)

class Disappearance( object ):
    def __init__(self, origin = Tracklet(), energy=None):
        self.origin = origin
        self.energy = energy
    def point_of_origin( self ):
        return (self.origin.x, self.origin.y, self.origin.z)

def move_from( prev_traxels, curr_traxels, move_ids, energy=None ):
    if(len(move_ids) !=2 ):
        raise Exception("moves_from(): move_ids has not length 2")
    origin = prev_traxels.with_id( move_ids[0] )
    if(len(origin.the) != 1):
        raise Exception("moves_from(): origin id not found or not unique")
    to = curr_traxels.with_id( move_ids[1] )
    if(len(to.the) != 1):
        raise Exception("moves_from(): to id not found or not unique")
    return Move(origin.the[0], to.the[0], energy)

def division_from( prev_traxels, curr_traxels, ids, energy=None ):
    if(len(ids) !=3 ):
        raise Exception("division_from(): ids has not length 3")
    origin = prev_traxels.with_id( ids[0] )
    if(len(origin.the) != 1):
        raise Exception("division_from(): origin id not found or not unique")
    to1 = curr_traxels.with_id( ids[1] )
    if(len(to1.the) != 1):
        raise Exception("division_from(): to1 id not found or not unique")
    to2 = curr_traxels.with_id( ids[2] )
    if(len(to2.the) != 1):
        raise Exception("division_from(): to2 id not found or not unique")

    return Division(origin.the[0], to1.the[0], to2.the[0], energy)

def appearance_from( curr_traxels, id, energy=None ):
    to = curr_traxels.with_id( id )
    if(len(to.the) != 1):
        raise Exception("appearance_from(): to id not found or not unique")
    return Appearance(to.the[0], energy)

def disappearance_from( prev_traxels, id, energy=None ):
    origin = prev_traxels.with_id( id )
    if(len(origin.the) != 1):
        raise Exception("disappearance_from(): origin id not found or not unique")
    return Disappearance(origin.the[0], energy)
