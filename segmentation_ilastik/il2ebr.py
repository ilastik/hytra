#!/usr/bin/env python
import h5py
import os.path as path
import sys



def ilastik2ebr( infn, outfn ):
    inf = h5py.File(infn, "r")
    outf = h5py.File(outfn, "w")

    raw = inf["volume/data"].value
    raw = raw[0,:,:,:,0]
    seg = inf["volume/labels/data"].value
    seg = seg[0,:,:,:,0]

    outf.create_dataset("raw/volume", data=raw, compression=1)
    outf.create_dataset("segmentation/gc", data=seg, compression=1)

    inf.close()
    outf.close()
    del inf
    del outf

if __name__ == "__main__":
    infns = sys.argv[1:]

    for infn in infns:
        outfn = path.basename(infn)
        print "-- processing", infn
        ilastik2ebr( infn, outfn )
