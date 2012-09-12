#!/usr/bin/env python

'''Convert time-lapse image stack to ilastik h5 file.'''

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

    in_tiffs = sorted(sys.argv[1:])
    out_fn = os.path.splitext(os.path.basename(in_tiffs[0]))[0] + "-" + os.path.splitext(os.path.basename(in_tiffs[-1]))[0] + ".h5"
    print out_fn
    f = h5py.File(out_fn,'w')
  
    ds = None
    for idx,t in enumerate(in_tiffs):
        nslices = vigra.impex.numberImages(t)
        v = [vigra.readImage( t , dtype='UINT8', index=i ) for i in range(nslices)]
        v = np.dstack(v)

        # convert to ilastik shape format (t,x,y,z,c)
        #a = v[np.newaxis,:,:,:, np.newaxis]
        a = v[:,:,:, np.newaxis]
        if idx == 0:
           shape = [len(in_tiffs)]
           shape.extend(a.shape)
           ds = f.create_dataset('/volume/data', shape, compression=1, dtype=np.uint8, chunks=(1,64,64,64,1))
           #ds = f.create_dataset('/volume/data', data=a, compression=1)
	
	ds[idx,...] = a

    f.close()
