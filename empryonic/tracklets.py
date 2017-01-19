from __future__ import unicode_literals
from builtins import next
from builtins import str
from builtins import object
import numpy as np
import copy
import unittest as ut


class Tracklet( object ):
    """A Tracklet may be a true cell, speckle or anything that can, but not necessarily should be tracked.

    By definition it is a point-like entity with three spatial, a temporal coordinate and an identifier. Additional features (like
    intensity) may be recorded in the 'meta' dictionary, that is an attribute of the class.
    """
    def __init__( self, x = 0,y = 0,z = 0,t = 0, id = 0):
        self.x = x
        self.y = y
        self.z = z
        self.t = t
        self.id = id
        self.meta = dict()

    def as_tuple( self ):
        '''(x, y, z, t, id)'''
        return (self.x, self.y, self.z, self.t, self.id)

    def __repr__( self ):
        return "".join([self.__class__.__name__,"(x = ",str(self.x),", y = ",str(self.y),", z = ",str(self.z),", t = ",str(self.t),", id = ",str(self.id),")"])

    def distance( self, tracklet ):
        '''Euclidean distance to other Tracklet.'''
        return np.linalg.norm(np.asarray((self.x, self.y, self.z)) - np.asarray((tracklet.x, tracklet.y, tracklet.z)))
        

    def nearest_neighbor( self, neighborhood ):
        '''Find nearest neighbor.
        
        neighborhood - collection of Traxels
        '''
        if not len(neighborhood) > 0:
            raise Exception("Tracklet::nearest_neighbor(): neighborhood is empty")
        
        it = iter(neighborhood)
        target = next(it)
        lowest_dist = self.distance(target)
        for neighbor in it:
            dist = self.distance(neighbor)
            if dist < lowest_dist:
                target = neighbor
                lowest_dist = dist
        return lowest_dist, target


class Tracklets( object ):
    def __init__( self, collection = [] ):
        self.the = collection

    def with_x( self, value ):
        return Tracklets([tr for tr in self.the if tr.x == value])
    def with_y( self, value ):
        return Tracklets([tr for tr in self.the if tr.y == value])
    def with_z( self, value ):
        return Tracklets([tr for tr in self.the if tr.z == value])
    def with_t( self, value ):
        return Tracklets([tr for tr in self.the if tr.t == value])
    def with_id( self, value ):
        return Tracklets([tr for tr in self.the if tr.id == value])

    def xy_swapped( self ):
        '''Iterator to the tracklets, swapping the x and y coordinate.'''
        for tr in self:
            swapped = copy.deepcopy(tr)
            temp = swapped.x
            swapped.x = swapped.y
            swapped.y = temp
            yield swapped
            
    def __str__( self ):
        return 'Collection of ' + str(len(self.the)) + ' tracklets'
    def __iter__( self ):
        return iter(self.the)
    def __len__( self ):
        return len(self.the)


class Test_Tracklets( ut.TestCase ):
    def setUp( self ):
        self.mocklets = Tracklets([Tracklet(1,2,3,0,23),Tracklet(1,2,4,1,19),Tracklet(4,5,6,2,27)])
    def test_xy_swapped( self ):
        swapped = self.mocklets.xy_swapped()
        for i, swapped_tr in enumerate(swapped):
            self.assertEqual(swapped_tr.x, self.mocklets.the[i].y)
            self.assertEqual(swapped_tr.y, self.mocklets.the[i].x)
            self.assertEqual(swapped_tr.z, self.mocklets.the[i].z)
            self.assertEqual(swapped_tr.id, self.mocklets.the[i].id)
            self.assertEqual(swapped_tr.t, self.mocklets.the[i].t)

    def test_with_id( self ):
        filtered = self.mocklets.with_id(19)
        self.assertEqual(filtered.the[0], self.mocklets.the[1])
        self.assertEqual(len(filtered.the), 1)

    def test_with_y( self ):
        filtered = self.mocklets.with_y(2)
        self.assertEqual(filtered.the[0], self.mocklets.the[0])
        self.assertEqual(filtered.the[1], self.mocklets.the[1])
        self.assertEqual(len(filtered.the), 2)

if __name__=="__main__":
    ut.main()
