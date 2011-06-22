/*
 * dFeatureExtractor.hxx
 *
 *  Created on: Mar 22, 2010
 *      Author: mlindner
 */

#ifndef DFEATUREEXTRACTOR_HXX_
#define DFEATUREEXTRACTOR_HXX_

#include <string>
#include <vector>
#include <cmath>

#include "vigra/hdf5impex.hxx"
#include "vigra/multi_array.hxx"
#include "vigra/matrix.hxx"
#include "vigra/eigensystem.hxx"


#include "supervoxelFeatures.hxx"
#include "readSuperVoxels.hxx"
#include "ConfigFeatures.hxx"

#include "config.hxx"

#include "vigra/timing.hxx"


namespace features {




typedef dSuperVoxelReader GeometryReader;

//typedef cgp::hdf5::GeometryReader<label_type, coordinate_type> GeometryReader;
//typedef GeometryReader::ThreeSet three_set;
//typedef vigra::TinyVector<coordinate_type,3> three_coordinate;


class dFeatureExtractor
{
public:
	dFeatureExtractor(std::string filename, bool verbose = false):
		r_(filename), filename_(filename), features_(FT_ALL), verbose_(verbose) {}
    dFeatureExtractor(std::string filename, feature_flag features, bool verbose = false):
		r_(filename), filename_(filename), features_(features), verbose_(verbose) {}

	// run feature extraction and write results to file
    std::string extract_features(feature_flag features);

	// change the features that will be extracted
    feature_flag enable_features(feature_flag features);
    feature_flag disable_features(feature_flag features);

private:
	std::string filename_;
    feature_flag features_;
	GeometryReader r_;
	bool verbose_;

	std::string rawdata_hdf5_path_();

	// get a set of coordinates of voxels, that belong to cell cell_number
    three_set getSuperVoxelCoordinates_(GeometryReader& r, label_type cell_number);
    three_set getLargeCoordinates_(GeometryReader& r, label_type cell_number);

	// extract the intensity values of a given coordinate set from the raw data volume
	feature_array getIntensityValues_(three_set& coordinate_set);

};


} /* namespace fextract*/

#endif /* DFEATUREEXTRACTOR_HXX_ */
