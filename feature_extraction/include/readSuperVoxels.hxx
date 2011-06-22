/*
 * readSuperVoxels.hxx
 *
 *  Created on: Apr 12, 2010
 *      Author: mlindner
 */

#ifndef READSUPERVOXELS_HXX_
#define READSUPERVOXELS_HXX_

#include <string>
#include <vector>

#include "vigra/hdf5impex.hxx"
//#include "hdf5impex.hxx"
#include "vigra/multi_array.hxx"

#include <ConfigFeatures.hxx>



class dSuperVoxelReader: public vigra::HDF5File
{
public:
	// new constructor
    dSuperVoxelReader(std::string filename): HDF5File(filename, vigra::HDF5File::Open){
		label_array max_labels (array_shape(4));
		read("/geometry/max-labels",max_labels);
		max_label_ = max_labels[3];

		three_counter_ = counter_type ( array_shape(max_label_), ulong (0));
		read("/geometry/parts-counters-3",three_counter_);
	}

	// read the list of coordinates for this label
    three_set threeSet(raw_type label);

	// return a list saying which labels are active
    label_array labelContent();

	// return the maximum voxel label (parameter "value" has no meaning for us; cgp would expect a 3 as parameter)
    label_type maxLabel(int value);

private:
	label_type max_label_;
	counter_type three_counter_;
};


#endif /* READSUPERVOXELS_HXX_ */
