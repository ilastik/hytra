#!/usr/bin/env python

'''Convert time-lapse image stack to ilastik h5 file.'''

import h5py
import vigra
import sys
import os
import os.path
import numpy as np
import optparse

if __name__ == '__main__':
   usage = """%prog [options] input-files
      Reads in ilastik segmentations from hdf5 files and outputs a 3d+t hdf5 file
   """
   parser = optparse.OptionParser(usage=usage)
   parser.add_option('-s', '--step', type='int', dest='step', default=1, help='include every n-th dataset [default: %default]')
   parser.add_option('--from', type='int', dest='from_ds', default=0, help='start with n-th dataset [default: %default]')
   parser.add_option('--max', type='int', dest='max_no', default=-1, help='include maximal n datasets; -1 for infinity [default: %default]')
   options, args = parser.parse_args()

   if len(args) > 0:
      in_fns = sorted(args)
   else:
      parser.print_help()
      sys.exit(1)

   step = options.step
   max_no = options.max_no
   from_ds = options.from_ds

   out_fn = os.path.splitext(os.path.basename(in_fns[from_ds]))[0] + \
      "-" + os.path.splitext(os.path.basename(in_fns[len(in_fns)-(len(in_fns)-1-from_ds)%step-1]))[0] + \
      '-step' + str(step) + ".h5"
   print out_fn
   f_out = h5py.File(out_fn,'w')
  
   ds_out = None
   count = 0
   for idx,fn in enumerate(in_fns[from_ds:]):
      if idx % step != 0:
         continue
      
      print fn
      f_in = h5py.File(fn,'r')
      ds_in = np.array(f_in['/volume/segmentation'],dtype=np.uint8)

      if ds_out is None:
         no_t = (len(in_fns)-from_ds)/step + (len(in_fns)-from_ds)%step
         if max_no > -1 and no_t > max_no:
            no_t = max_no
         shape = [no_t]
         shape.extend(ds_in.shape[1:])  # all except first dimension
         ds_out = f_out.create_dataset('/volume/segmentation', shape, compression=1, dtype=np.uint8, chunks=(1,64,64,64,1))
   
      ds_out[count,...] = ds_in
      f_in.close()

      count += 1
      if max_no > -1 and count >= max_no:
         break

   f_out.close()

