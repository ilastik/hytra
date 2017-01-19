from __future__ import print_function
from __future__ import unicode_literals
#
# (c) Martin Lindner, 2010
#

from builtins import range
import os
import time
from scipy import io
import sys
import numpy as np
import h5py

def convert( filename, h5filename, matname = 'stack', group = '/raw', dataset = 'volume', compr = 0 ):
	"""Convert a digital embryos Matlab file to HDF5 format.
	
	Compress the dataset, if requested

	"""
	#load .mat file
	mat_content = io.loadmat( filename, struct_as_record=True)
	
	#create HDF5 file and group
	h5file = h5py.File( h5filename, 'w' )
	rawgroup = h5file.create_group( group )
	if compr > 0:
		#set chunk size and save compressed
		cs = min(200,min(mat_content[matname].shape))
		csize = (cs, cs, cs)
		rawgroup.create_dataset( dataset, data=mat_content[matname].astype('uint16'), dtype=np.uint16, chunks=csize, compression='gzip', compression_opts = compr)
	else:
		#save without compression
		rawgroup.create_dataset( dataset, data=mat_content[matname].astype('uint16'), dtype=np.uint16)
	
	h5file.flush()
	h5file.close()
	
	

def compare( file1, file2, matname = 'stack', dataset = '/raw/volume' ):
	"""Check, if two volumes in two files contain the same data.

	The files can both be Matlab or HDF5 files.

	"""
	f1isHdf5 = False
	f2isHdf5 = False
	#check if file 1 is Matlab or HDF5
	if file1[-4:] == '.mat':
		matfile1 = io.loadmat( file1, struct_as_record=True)
		data1 =  matfile1[matname]
	else:
		h5file1 = h5py.File( file1, 'r' )
		data1 = h5file1[dataset]
		f1isHdf5 = True
	
	#check if file s is Matlab or HDF5
	if file2[-4:] == '.mat':
		matfile2 = io.loadmat( file2, struct_as_record=True)
		data2 =  matfile2[matname]
	else:
		h5file2 = h5py.File( file2, 'r' )
		data2 = h5file2[dataset]

	#check if arrays are equal
	ret = np.array_equal(data1, data2)
	
	# close HDF5 files if they were opened
	if f1isHdf5:
		h5file1.close()
	if f2isHdf5:
		h5file2.close()
	
	return ret



def batchConvert( folder , compression = 0):
	"""Convert Matlab (.mat) files into HDF5 datasets.

	All Matlab files in the given folder will be processed.
	The Matlab files are expected to contain a volume named 'stack'.
	The data will be written to '/raw/volume' inside the HDF5 file. 

	"""
	t0 = time.time()
	conflicts = 0
	total = 0
	files = os.listdir( folder )
	for i in range(0, len(files)):
		if files[i][-3:] == 'mat':
			total=total+1
			oldname = os.path.join(folder,files[i])

			(newname, ext) = os.path.splitext(files[i])
			newname = os.path.join(folder,newname + '.h5')
			
			convert(oldname, newname, compr = compression)
			if not compare(oldname, newname):
				conflicts=conflicts+1
				print('Error: Copying failed: "', oldname, '" is not equal to "', newname, '"!')
	
	print('Conversion finished! ', total, ' files processed, ', conflicts, 'conflicts detected.')
	print((time.time() - t0), "seconds wall time")		

	
if __name__ == "__main__":
	batchConvert( sys.argv[1], compression = 4)
