#!/usr/bin/env python

import sys
import os.path as path
import vigra
import numpy as np
import h5py
from tracking_io import LineageH5
import optparse
import glob


def edit_path(folder):
    if folder.split("/")[-1] == '':
        return folder
    else:
        return folder + "/"


def relabel( volume, replace ):
    mp = np.arange(0,np.amax(volume)+1, dtype=volume.dtype)
    mp[1:] = 255
    mp[replace.keys()] = replace.values()
    return mp[volume]

if __name__ == "__main__":
    usage = """%prog [options] H5 File"""
    parser = optparse.OptionParser(usage=usage)
    
    parser.add_option('-p', '--prefix', type='string', dest='prefix', default='0000', help='output files prefix [default: %default]')
    parser.add_option('-o', type='string', dest='out_dir', default='.', help='output directory [default: %default]')
    
    options, args = parser.parse_args()

    numArgs = len(args)
    if numArgs > 0:
        fns = []
        for arg in args:
            fns.extend(glob.glob(arg))
    else:
        parser.print_help()
        sys.exit(1)

    
    print "-- mapping to colors"

    label2color = []
    label2color.append({})

    
    
    for fn in fns[1:]:

        label2color.append({})
        with LineageH5(fn, 'r') as f:
            print path.basename(fn)
            
        
            #for dis in f.get_disappearances():
                
            #    label2color[-2][dis[0]] = 255 # mark disapps
            

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
            

    print "-- creating colored arrays"
    for idx, fn in enumerate(fns):
        with h5py.File(fn, 'r') as f:
            print path.basename(fn)
            im = f['segmentation/labels'].value
            
            relabeled = relabel(im, label2color[idx])
            for i in range(0, im.shape[2]):
                out_im = relabeled[:, :, i]

                

                out_fn = edit_path(options.out_dir) + options.prefix + "%04d_vis_" % (idx) + path.basename(fn) + "%04d.tif" % (i)
                
                vigra.impex.writeImage(np.asarray(out_im,dtype=np.uint8), out_fn)
    
