#!/usr/bin/env python

import argparse
import os
import os.path as path
import h5py
import vigra
import numpy
import libdetection as fun
import gc, sys, cPickle, os.path
import ilastik.core.jobMachine

''' objects format

objects = [object,...]
object = xs, ys, zs # all the voxels constituting the object
xs = list of x coordinates of the voxels
len(xs) == len(ys) == len(zs)

'''

def process(h5file, synapselabel, threshold, mingoodsize, maxgoodsize, outputfile):
    print "== processing " + os.path.basename(h5file)
    threshref = [threshold/(1-threshold), 1.0]
    thresh = [threshold/(1-threshold), 1.0]

    thresholds = []
    thresholds.append(threshref)
    thresholds.append(thresh)

    print "-- searching objects"
    objsref, objsuser, volume_shape = fun.findObjectsNew(h5file, synapselabel, thresholds)
    print "-> %d objects found" % len(objsref)
    print "-> %d objects selected" % len(objsuser)
    print

    print "-- filtering selected objects"
    print "-- (min. size: %d, max. size: %d)" % (mingoodsize, maxgoodsize)
    objs_final, bboxes_final = fun.filterObjects3d(objsuser, objsref, mingoodsize, maxgoodsize)
    print "-> %d objects passed filter" % len(objs_final)
    print

    print "-- saving results"
    foutHandle = h5py.File(outputfile, "w")

    volume = foutHandle.create_group('volume')
    dset = volume.create_dataset("data", (volume_shape[0], volume_shape[1], volume_shape[2], volume_shape[3], 1), "uint8")
    print "writing to file", outputfile

    f2 = h5py.File(h5file, "r")
    dset[:] = f2["/volume/data"][:]

    print "data created, creating labels"
    lg = volume.create_group('labels')
    newlabels_data = numpy.zeros((volume_shape[0], volume_shape[1], volume_shape[2], volume_shape[3], 1), dtype = "uint8")
    for ii, obj in enumerate(objs_final):
        for iii in range(len(obj[0])):
            newlabels_data[0, obj[0][iii], obj[1][iii], obj[2][iii], 0] = 1

    newlabels = lg.create_dataset("data", data=newlabels_data)
    lg.attrs['color'] = [4294967040]
    lg.attrs['name'] = ["object"]
    lg.attrs['number'] = [1]    

    print "streaming...",
    foutHandle.close()
    gc.collect()
    print "done!"
    print

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract objects from probability maps.')
    parser.add_argument('--threshold', '-t', type=float, default=0.5, help='minimal probability (default: %(default)s)')
    parser.add_argument('--minsize', type=int, default=0, help='minimal voxel number (default: %(default)s)')
    parser.add_argument('--maxsize', type=int, default=100000, help='maximal voxel number (default: %(default)s)')
    parser.add_argument('--label', type=int, default=0, help='label corresponding to the object class (default: %(default)s)')
    parser.add_argument('-o', default='./objects', help='output directory (default: %(default)s)')
    parser.add_argument('ih5s', nargs='+', help='files containing probability maps (ilastik h5 format)')

    if(len(sys.argv) == 1):
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    # output dir
    if(not path.exists(args.o)):
        os.mkdir(args.o)

    def func(fn):
        process(fn, args.label, args.threshold, args.minsize, args.maxsize, path.join(args.o, "objects_" + path.basename(fn)))

    from multiprocessing import Pool
    p = Pool()
    p.map(func, args.ih5s)
    
ilastik.core.jobMachine.GLOBAL_WM.stopWorkers()
del ilastik.core.jobMachine.GLOBAL_WM
ilastik.core.jobMachine.GLOBAL_WM = None
