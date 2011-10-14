#!/usr/bin/env python

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


'''
if len(sys.argv) != 7:
    print len(sys.argv)
    print 'Usage: synapse_detection_script_final.py h5_processed_file_path synapse_label_number threshold min_synapse_size max_synapse_size outputfile(optional)'
    print 'For example: python synapse_detection_script.py test1.h5_processed.h5 2 9.0 1000 250000 test1_output.h5'
    print '\n'
    #ilastik.core.jobMachine.GLOBAL_WM.stopWorkers()
    #del ilastik.core.jobMachine.GLOBAL_WM
    #ilastik.core.jobMachine.GLOBAL_WM = None
    sys.exit(1)

 
h5file = sys.argv[1]
synapselabel = int(sys.argv[2])
threshold = float(sys.argv[3])
mingoodsize = int(sys.argv[4])
maxgoodsize = int(sys.argv[5])
outputfile = sys.argv[6]

assert(threshold >= 0. and threshold < 1.)
'''

synapselabel = 0
threshold = 0.4
mingoodsize = 600
maxgoodsize = 100000

fn = "/home/bkausler/data/hufnagel_2011-10-06_drosophila-syncytial-blastoderm/experiment1/classified/Time0000%02d_00.h5_processed.h5" 
outfn = "/home/bkausler/data/hufnagel_2011-10-06_drosophila-syncytial-blastoderm/experiment1/segmented/%02d.h5"

def func(i):
    process(fn %i, synapselabel, threshold, mingoodsize, maxgoodsize, outfn % i)

if __name__ == "__main__":
    from multiprocessing import Pool
    p = Pool(4)
    p.map(func, range(60,100))
    
ilastik.core.jobMachine.GLOBAL_WM.stopWorkers()
del ilastik.core.jobMachine.GLOBAL_WM
ilastik.core.jobMachine.GLOBAL_WM = None
