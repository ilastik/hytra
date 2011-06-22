/*
 * readSuperVoxels.cxx
 *
 *  Created on: Apr 12, 2010
 *      Author: Martin Lindner
 */

#include "readSuperVoxels.hxx"



three_set dSuperVoxelReader::threeSet(raw_type label)
{
	three_set coordinates;

	// check if label is out of bounds
	if(label > max_label_ || label <= 0){
		std::cerr << "dSuperVoxelReader::threeSet(): label out of bounds. Allowed: 0 < label < " << max_label_ << ", supplied: " << label << ".\n";
		return coordinates;
	}

	// check if there is a supervoxel for this label
	if(three_counter_[label-1] == 0){
		return coordinates;
	}


	// set up path to dataset
	char path [100];
	sprintf(path,"/geometry/3-sets/bin-%i/%i-1", label, label);

	// prepare vigra array
    //hssize_t num_dimensions = getDatasetDimensions(path);
    vigra::ArrayVector<hsize_t> dim_shape = getDatasetShape(path);
    //get_dataset_size(path,num_dimensions,dim_shape);
	set_type readData (set_shape(dim_shape[0], dim_shape[1]));

	// read dataset
	read(path, readData);

	// convert data
	hsize_t size = dim_shape[0];

	for(hsize_t i = 0; i < size; i++){
		three_set::value_type el (readData(i,0), readData(i,1), readData(i,2));
		coordinates.push_back(el);
	}

	return coordinates;
}


label_array dSuperVoxelReader::labelContent(){
	label_array content = label_array(array_shape(max_label_));
	read("/geometry/parts-counters-3", content);

	return content;
}


label_type dSuperVoxelReader::maxLabel(int value = 3)
{
	return max_label_;
}

