import sys

import vigra
import numpy
import h5py
from ilastik.core.overlays import thresholdOverlay
from ilastik.modules.connected_components.core import connectedComponentsMgr
from ilastik.core.volume import VolumeLabels
from ilastik.core.volume import DataAccessor
from ilastik.core import dataMgr
import time
import gc
import os.path

class dummyOverlay:
    def __init__(self, data, color):
        self._data = data
        self._data = self._data.reshape(self._data.shape+(1,))
        self.color = color

def objectsSlow2d(cc):
    ncomp = numpy.amax(cc)
    objs = {}
    for ic in range(ncomp+1):
        objs[ic] = [[], []]
        
    for i in range(cc.shape[1]):
        for j in range(cc.shape[2]):
            ind = int(cc[0, i, j, 0])
            if ind!=0:
                objs[ind][0].append(i)
                objs[ind][1].append(j)

    return objs
    
    
def objectsSlow3d(cc, mingoodsize = None, maxgoodsize = None, outputfile=None):

    ncomp = numpy.amax(cc)
    objs = []

    for ic in range(ncomp+1):
        objs.append([[], [], []])

    nzindex = numpy.nonzero(cc)
    for i in range(len(nzindex[0])):
        #value = cc[nzindex[0][i], nzindex[1][i], nzindex[2][i], 0]
        value = cc[nzindex[0][i], nzindex[1][i], nzindex[2][i]]
        if value > 0:
            objs[value][0].append(nzindex[0][i])
            objs[value][1].append(nzindex[1][i])
            objs[value][2].append(nzindex[2][i])
            
    return objs



def objects_from( threshold_overlay, threshold ):
    assert(len(threshold) == 2)

    threshold_overlay.setThresholds(threshold)
    accessor = thresholdOverlay.MultivariateThresholdAccessor(threshold_overlay)
    data = numpy.asarray(accessor[0, :, :, :, 0], dtype='uint8')
    data = data.swapaxes(0,2).view()
    cc = vigra.analysis.labelVolumeWithBackground(data, 6, 2)
    cc = cc.swapaxes(0,2).view()
    return objectsSlow3d(cc)



def findObjectsNew(h5file, synapselabel, thresh):
    assert(len(thresh) == 2)

    f2 = h5py.File(h5file, "r")
    pred = f2["/volume/prediction"].value
    print "loading prediction for object class...",
    sys.stdout.flush()
    foreground = dummyOverlay(pred[:, :, :, :, synapselabel], synapselabel+1)
    print "done!"

    backgrounds = []
    print "loading prediction for background class(es)...",
    sys.stdout.flush()
    for i in range(pred.shape[4]):
            #collect background labels
            if i!=synapselabel:
                label = dummyOverlay(pred[:, :, :, :, i], i+1)
                backgrounds.append(label)
    print "done!"

    print "init thresholding...",
    sys.stdout.flush()
    th_over = thresholdOverlay.ThresholdOverlay([foreground], backgrounds, 5)
    print "done!"
    print "extracting objects"
    sys.stdout.flush()
    objs_ref = objects_from( th_over, thresh[0])
    print "done!"
    print "extracting selection objects"
    sys.stdout.flush()
    objs_user = objects_from( th_over, thresh[1])
    print "done!"
    return (objs_ref, objs_user, pred.shape)
    
def findObjects(foreground, backgrounds, thresh):
    th_over = thresholdOverlay.ThresholdOverlay([foreground], backgrounds, 5)
    th_over.setThresholds(thresh)
    accessor = thresholdOverlay.MultivariateThresholdAccessor(th_over)
    data = numpy.asarray(accessor[0, :, :, :, 0], dtype='uint8')
    data = data.swapaxes(0,2).view()
    cc = vigra.analysis.labelVolumeWithBackground(data, 6, 2)
    cc = cc.swapaxes(0,2).view()
    objs = []
    if cc.shape[0]>1:
        #3d data
        objs = objectsSlow3d(cc)
    else:
        #2d data
        objs = objectsSlow2d(cc)
    return objs

def findObjectsFile(thresh_file):
    print "thfile: ", thresh_file
    fin = h5py.File(thresh_file, "r")
    data = numpy.array(fin["/volume/data"])
    data = data[0, :, :, :, 0]
    
    data = data.swapaxes(0,2).view()
    res = vigra.analysis.labelVolumeWithBackground(data, 6, 2)
    res = res.swapaxes(0, 2).view()
    objs = objectsSlow3d(res)
    fin.close()
    return objs
    
    
#def filter_by_size(objs, minsize, maxsize):
#   return (obj for obj in objs if 
    
def filterObjects3d(objssmall, objsref, mingoodsize, maxgoodsize):
    '''Filter connected components

    first, it finds all objects greater than mingoodsize
    then, it loops through reference objects and only takes the ones that contain
    an object from the other list inside. 
    last step throws out the too big ones.
    
    objsref -- filter these objects
    '''
    goodobjs = []
    bboxes = []
    print "filtering..."
    for i, value in enumerate(objssmall):
        if len(value[0])>mingoodsize:
            goodobjs.append(value)
            bboxes.append([numpy.amin(value[0]), numpy.amin(value[1]), numpy.amin(value[2]), numpy.amax(value[0]), numpy.amax(value[1]), numpy.amax(value[2])])
    goodobjsref = []
    bboxesref = []

    for value in objsref:
        if len(value[0])>mingoodsize:
            goodobjsref.append(value)
            bboxesref.append([numpy.amin(value[0]), numpy.amin(value[1]), numpy.amin(value[2]), numpy.amax(value[0]), numpy.amax(value[1]), numpy.amax(value[2])])
    #print "after size filtering, with user threshold ", len(goodobjs), ", with ref. threshold ", len(goodobjsref)
    found = False
    foundlist = []
    objs_final = []
    bboxes_final = []
    for i, big in enumerate(bboxesref):
        found = False
        for j, small in enumerate(bboxes):
            #if big[0]>small[3] or small[0]>big[3] or big[1]>small[4] or small[1]>big[4] or big[2]>small[5] or small[2]>big[5]:
            if small[0]>=big[0] and small[3]<=big[3] and small[1]>=big[1] and small[4]<=big[4] and small[2]>=big[2] and small[5]<=big[5]:
                found = True
                break
                
            else:
                continue    
        if found == True:
            if len(goodobjsref[i][0])<maxgoodsize:
                objs_final.append(goodobjsref[i])
                bboxes_final.append(big)
    print "total synapses found: ", len(objs_final)        
    return objs_final, bboxes_final

def blockwiseThreshold(h5file, synapselabel, thresh, outputfile, blocksize, smoothing_sigma):
    f2 = h5py.File(h5file, "r")
    pred = f2["/volume/prediction"]
    print pred.shape
    shape_to_return = pred.shape
    mpa = dataMgr.MultiPartDataItemAccessor(DataAccessor(pred), blocksize, smoothing_sigma*2)
    #save thresholded overlays into files
    outfiles = []
    outhandles = []
    outgroups = []
    for t in thresh:
        t_to_print = round(t[0]/t[1], 1)
        outfile_base = os.path.splitext(outputfile)[0]
        outfile = outfile_base + "_th_" + str(t_to_print)+".h5"
        print "saving thresholded predictions into ", outfile
        outfiles.append(outfile)
        fth = h5py.File(outfile, "w")
        outhandles.append(fth)
        gth = fth.create_group("volume")
        outgroups.append(gth)
        
    for blockNr in range(mpa.getBlockCount()):
        print "Part " + str(blockNr) + "/" + str(mpa.getBlockCount()) + " "
        dm = dataMgr.DataMgr()
        di = mpa.getDataItem(blockNr)
        foreground = dummyOverlay(di[:, :, :, :, synapselabel], synapselabel+1)
        backgrounds = []
        for i in range(di.shape[4]):
            #collect background labels
            if i!=synapselabel:
                label = dummyOverlay(di[:, :, :, :, i], i+1)
                backgrounds.append(label)
        th_over = thresholdOverlay.ThresholdOverlay([foreground], backgrounds, smoothing_sigma)
        for ith, t in enumerate(thresh):
            print t
            th_over.setThresholds(t)
            accessor = thresholdOverlay.MultivariateThresholdAccessor(th_over)
            destbegin = di._writeBegin
            destend =  di._writeEnd
            srcbegin =  di._readBegin
            srcend =  di._readEnd
            destshape = di._writeShape
            vl = VolumeLabels(accessor)
            vl._data.serialize(outgroups[ith], "data", destbegin, destend, srcbegin, srcend, destshape)
            gc.collect()
    for fth in outhandles:    
        fth.close()
    f2.close()
    return shape_to_return, outfiles



    



