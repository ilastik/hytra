from __future__ import print_function
from __future__ import unicode_literals
import itertools
import sys
import numpy as np

def makeMaxIntensityProjector( axis ):
    return lambda volume: volume.max(axis = axis).astype(np.float32)

def makeMeanIntensityProjector( axis ):
    return lambda volume: volume.mean(axis = axis).astype(np.float32)

def spacetime( volumes, projector = makeMaxIntensityProjector(2)):
    '''Map 4D data into a stack of images.

    An intensity projection is employed to map a volume into a
    two-dimensional image. A stack of these images is returned as
    a numpy volume in order of the input volumes.
    
    volumes: a sequence of three-dimensional numpy arrays of scalars
    projector: function :: volume -> 2D matrix
    '''
    class verboseProjector:
        def __init__( self ):
            self.count = 0
        def __call__( self, volume ):
            self.count += 1
            print("Processing volume #" + str(self.count))
            sys.stdout.flush()
            return projector( volume )

    
    return np.dstack(itertools.imap(verboseProjector(), volumes))
