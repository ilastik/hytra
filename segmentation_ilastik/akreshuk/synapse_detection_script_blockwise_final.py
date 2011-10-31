import h5py
import vigra
import numpy
import synDetectionHelperFunctions as fun
import gc, sys, cPickle, os.path
import ilastik.core.jobMachine

##############################
# This script searches for synapses by first thresholding
# the predictions in a blockwise manner and writing the 
# resulting thresholded data into temporary files
# Use it if you don't have enough RAM to load the full
# processed dataset.
# The size of the blocks is set in the next 2 lines. Feel free to change it.
##############################

blocksize = 512
smoothing_sigma = 5 

if len(sys.argv) != 6 and len(sys.argv) != 7:
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
if len(sys.argv)>6:
    outputfile = sys.argv[6]
else:
    outputfile = h5file

threshref = [0.5, 0.5]
if threshold>100:
    print "Threshold is a bit too high. Try to get it down to under 100"
    sys.exit(1)

thresh = [threshold/100., 0.01]

thresholds = []
thresholds.append(threshref)
thresholds.append(thresh)
    
volume_shape, th_files = fun.blockwiseThreshold(h5file, synapselabel, thresholds, outputfile, blocksize, smoothing_sigma)

gc.collect()

#find the synapse objects
objsref = []
objsuser = []
outfiles = []

print "finding objects..."
for th_file in th_files:
    if th_file.endswith("_th_1.0.h5"):
        objsref = fun.findObjectsFile(th_file)
    else:
        objs = fun.findObjectsFile(th_file)
        objsuser.append(objs)
        th_file_dir = os.path.split(th_file)[0]
        th_file_name = os.path.split(th_file)[1]
        th_file_base = os.path.splitext(th_file_name)[0]
        parts = th_file_base.split('_')
        th_value = parts[-1]
        outfiles.append(th_file_dir + "/" + th_file_base + "_synapses.h5")
  
    gc.collect()
    
for i, objs in enumerate(objsuser):
    objs_final, bboxes_final = fun.filterObjects3d(objs, objsref, mingoodsize, maxgoodsize)
    
    foutHandle = h5py.File(outfiles[i], "w")
    
    volume = foutHandle.create_group('volume')
    dset = volume.create_dataset("data", (volume_shape[0], volume_shape[1], volume_shape[2], volume_shape[3], 1), "uint8")
    print "writing to file", outfiles[i]
    
    
    f2 = h5py.File(h5file)
    dset[:] = f2["/volume/data"][:]
    
    print "data created, creating labels"
    lg = volume.create_group('labels')
    newlabels_data = numpy.zeros((volume_shape[0], volume_shape[1], volume_shape[2], volume_shape[3], 1), dtype = "uint8")
    for ii, obj in enumerate(objs_final):
        print "streaming object ", ii, "/", str(len(objs_final))
        for iii in range(len(obj[0])):
            newlabels_data[0, obj[0][iii], obj[1][iii], obj[2][iii], 0] = 1
            
    newlabels = lg.create_dataset("data", data=newlabels_data)
    lg.attrs['color'] = [4294967040]
    lg.attrs['name'] = ["synapse"]
    lg.attrs['number'] = [1]    

    print "streaming..."
    foutHandle.close()
    
    print "done!"
    
ilastik.core.jobMachine.GLOBAL_WM.stopWorkers()
del ilastik.core.jobMachine.GLOBAL_WM
ilastik.core.jobMachine.GLOBAL_WM = None
    
