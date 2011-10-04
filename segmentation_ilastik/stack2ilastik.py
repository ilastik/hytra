#!/usr/bin/env python

'''Convert image stack to ilastik h5 file.'''

import h5py
import vigra
import sys
import numpy as np

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print "Arguments: in_filname (one file of the stack to be loaded) [out_filename] (default: out.h5)"
        sys.exit(1)

    in_fn = sys.argv[1]
    out_fn = sys.argv[2] if len(sys.argv) > 2 else 'out.h5'

    v = vigra.readVolume( in_fn , dtype='UINT8')

    # convert to ilastik shape format (t,x,y,z,c)
    a = v[np.newaxis,:]

    f = h5py.File(out_fn, 'w')
    f.create_dataset('/volume/data', data=a, compression=1)
    
