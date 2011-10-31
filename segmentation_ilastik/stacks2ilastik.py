#!/usr/bin/env python

'''Convert image stacks to ilastik h5 file.'''

import h5py
import vigra
import sys
import os
import os.path
import numpy as np

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print "Arguments: multipage-tiffs"
        sys.exit(1)

    in_tiffs = sys.argv[1:]
    out_fn = "stacks.h5"
    stacks = []
    for t in in_tiffs:


        print out_fn
        nslices = vigra.impex.numberImages(t)
        v = [vigra.readImage( t , dtype='UINT8', index=i ) for i in range(nslices)]
        stacks.extend(v)
        

    # convert to ilastik shape format (t,x,y,z,c)
    stacks = np.dstack(stacks)
    a = stacks[np.newaxis,:,:,:, np.newaxis]
    print a.shape
    f = h5py.File(out_fn, 'w')
    f.create_dataset('/volume/data', data=a, compression=1)
