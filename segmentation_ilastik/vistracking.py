#!/usr/bin/env python

import sys
import os.path as path
import vigra
import numpy as np
import h5py
from empryonic.io import LineageH5

def relabel( volume, replace ):
    mp = np.arange(0,np.amax(volume)+1, dtype=volume.dtype)
    mp[1:] = 255
    mp[replace.keys()] = replace.values()
    return mp[volume]


fns = sys.argv[1:]

label2color = []

##
## map to colors
##
print "-- mapping to colors"
# init first timestep with random colors
label2color.append({})
#with LineageH5(fns[0], 'r') as f:
#    print path.basename(fns[0])
#    labels = f['objects/meta/id'].value
#    for label in labels:
#        color = np.random.randint(2,255)
#        label2color[-1][label] = color

# map the rest
for fn in fns[1:]:
    label2color.append({})
    with LineageH5(fn, 'r') as f:
        print path.basename(fn)
        labels = f['objects/meta/id'].value

        for dis in f.get_disappearances():
            label2color[-2][dis[0]] = 255 # mark disapps

        for app in f.get_appearances():
            label2color[-1][app[0]] = np.random.randint(1,255)

        for move in f.get_moves():
            if not label2color[-2].has_key(move[0]):
                label2color[-2][move[0]] = np.random.randint(1,255)
            label2color[-1][move[1]] = label2color[-2][move[0]]

        for division in f.get_divisions():
            if not label2color[-2].has_key(division[0]):
                label2color[-2][division[0]] = np.random.randint(1,255)
            ancestor_color = label2color[-2][division[0]]
            label2color[-1][division[1]] = ancestor_color
            label2color[-1][division[2]] = ancestor_color
    
##
## write out 'colored' arrays
##
print "-- creating colored arrays"
for idx, fn in enumerate(fns):
    with h5py.File(fn, 'r') as f:
        print path.basename(fn)
        proj = f['segmentation/labels'].value.max(axis=2)
        relabeled = relabel(proj, label2color[idx])
    vigra.impex.writeImage(np.asarray(relabeled,dtype=np.uint8), "vis_"+ path.basename(fn)+".tif")
    #with h5py.File("vis_"+ path.basename(fn), 'w') as f_out:
    #    f_out.create_dataset('connected_components', data=relabeled, dtype=np.uint8, compression=2)
