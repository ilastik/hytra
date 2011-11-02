#!/usr/bin/env python
import h5py
import sys
import vigra
import numpy as np

def interpolate( fn ):
    print "Interpolating:", fn
    datasetn = "/volume/prediction"
    interp_dsn = datasetn + "_interp"

    f = h5py.File(fn, 'a')
    print "Loading dataset...",
    d = f[datasetn].value
    print "done!"

    print "Processing..."
    print "Old shape", d.shape
    new_shape = (d.shape[1], d.shape[2], int(d.shape[3] * 12.3))
    sub = d[0,...]
    interp = vigra.resize(sub, shape= new_shape)
    interp = interp[np.newaxis,...]
    print "New shape", interp.shape    
    del d
    print "Interpolation done!"
    print "Writing..."
    sys.stdout.flush()
    f.create_dataset(interp_dsn, data=interp, compression=1)
    f.close()
    del interp
    print "done:", fn

if __name__ == '__main__':
    import multiprocessing as mp
    p = mp.Pool(1)
    p.map(interpolate, sys.argv[1:])
