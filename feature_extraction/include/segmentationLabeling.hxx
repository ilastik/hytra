/*
 * segmentationLabeling.hxx
 *
 *  Created on: Apr 22, 2010
 *      Author: mlindner
 */

#ifndef SEGMENTATIONLABELING_HXX_
#define SEGMENTATIONLABELING_HXX_

#include <string>
#include <map>

#include "vigra/hdf5impex.hxx"
#include "vigra/multi_array.hxx"
#include "vigra/labelvolume.hxx"
#include "vigra/union_find.hxx"
#include "vigra/functorexpression.hxx"
#include "vigra/timing.hxx"

#include "config.hxx"
#include "ConfigFeatures.hxx"

// iterators for the segmentation volume
typedef seg_volume::traverser iter3_t;
typedef iter3_t::next_type iter2_t;
typedef iter2_t::next_type iter1_t;

int segmentationLabeling(FileConfiguration fc, TaskConfiguration tc);
int segmentationLabelingAtOnce(FileConfiguration fc, TaskConfiguration tc);

//void shiftLabels(seg_volume &test_volume, label_type max_label);
void findUnions(seg_volume &test_volume, seg_volume &ref_slice,  vigra::detail::UnionFindArray<label_type> &label_map );
void checkBorders(std::string labelsPath, seg_volume &test_volume, vigra::HDF5File &seg_file, coordinate_type ol0, coordinate_type ol1, coordinate_type ol2,
        volume_shape &block_offset, volume_shape &block_shape, vigra::detail::UnionFindArray<label_type> &label_map );
int relabel(seg_volume &test_volume, volume_shape &block_offset, vigra::detail::UnionFindArray<label_type> &label_map,
             std::map<label_type,three_set> &supervoxels);

#endif
