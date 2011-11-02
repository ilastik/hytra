/*
 * segmentationLabeling
 *
 *  Created on: Apr 22, 2010
 *      Author: Martin Lindner
 */

#include "segmentationLabeling.hxx"


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------


// obsolete since using functor
/*
void shiftLabels(seg_volume &test_volume, label_type max_label){
    // replace old labels to get correct labeling over whole volume
    iter3_t i3_f = test_volume.traverser_begin();
    iter3_t i3_l = test_volume.traverser_end();
    // iterate over the third dimension
    for ( ; i3_f != i3_l; ++i3_f){
        iter2_t i2_f = i3_f.begin ();
        iter2_t i2_l = i3_f.end ();
        // iterate over the second dimension
        for ( ; i2_f != i2_l; ++i2_f){
            iter1_t i1_f = i2_f.begin ();
            iter1_t i1_l = i2_f.end ();
            // iterate over the first dimension
            for (; i1_f != i1_l; ++i1_f){
                // replace old label (except background)
                label_type test_label = *i1_f;
                if(test_label != 0){
                    *i1_f = test_label + max_label;
                }
            }
        }
    }

    return ;
}
*/

// go through the borders of the block test_volume and compare the labels to
// labels in ref_slice. If they differ, unify them.
void findUnions(
        seg_volume &test_volume, // volume with potentially wrong labels
        seg_volume &ref_slice, // reference labels
        vigra::detail::UnionFindArray<label_type> &label_map // labels to be replaced
        ){
    // collect wrong labels
    iter3_t it3_f = test_volume.traverser_begin();
    iter3_t ir3_f = ref_slice.traverser_begin();
    iter3_t ir3_l = ref_slice.traverser_end();
    // iterate over the third dimension
     for ( ; ir3_f != ir3_l; ++ir3_f, ++it3_f){
         iter2_t ir2_f = ir3_f.begin ();
         iter2_t ir2_l = ir3_f.end ();
         iter2_t it2_f = it3_f.begin ();
         // iterate over the second dimension
         for ( ; ir2_f != ir2_l; ++ir2_f, ++it2_f){
             iter1_t ir1_f = ir2_f.begin ();
             iter1_t ir1_l = ir2_f.end ();
             iter1_t it1_f = it2_f.begin ();
             // iterate over the first dimension
             for (; ir1_f != ir1_l; ++ir1_f, ++it1_f){
                 // test if labels differ
                label_type test_label = *it1_f;
                label_type ref_label = *ir1_f;
                if(ref_label != test_label && test_label != 0 && ref_label != 0){
                    label_map.makeUnion(ref_label, test_label);
                }
             }
         }
     }
     return;
}


void checkBorders(
        std::string labelsPath,
        seg_volume &test_volume,  // labeled volume to be tested for multiple labeling
        vigra::HDF5File &seg_file, // file handle, load slices to compare
        coordinate_type ol0, coordinate_type ol1, coordinate_type ol2, // overlap
        volume_shape &block_offset, volume_shape &block_shape, // current block parameters
        vigra::detail::UnionFindArray<label_type> &label_map // labels to be replaced
        ){

    if(ol0 == 1){
        // load the overlap slice
        volume_shape overlap_shape (1, block_shape[1], block_shape[2]);
        volume_shape overlap_offset (block_offset[0]-1, block_offset[1], block_offset[2]);
        seg_volume overlap (overlap_shape);
        seg_file.readBlock(labelsPath, overlap_offset, overlap_shape, overlap);

        // go through overlap slice, compare and collect labels
        findUnions(test_volume, overlap, label_map);
    }
    if(ol1 == 1){
        // load the overlap slice
        volume_shape overlap_shape (block_shape[0], 1, block_shape[2]);
        volume_shape overlap_offset (block_offset[0], block_offset[1]-1, block_offset[2]);
        seg_volume overlap (overlap_shape);
        seg_file.readBlock(labelsPath, overlap_offset, overlap_shape, overlap);

        // go through overlap slice, compare and collect labels
        findUnions(test_volume, overlap, label_map);
    }
    if(ol2 == 1){
        // load the overlap slice
        volume_shape overlap_shape (block_shape[0], block_shape[1], 1);
        volume_shape overlap_offset (block_offset[0], block_offset[1], block_offset[2]-1);
        seg_volume overlap (overlap_shape);
        seg_file.readBlock(labelsPath, overlap_offset, overlap_shape, overlap);

        // go through overlap slice, compare and collect labels
        findUnions(test_volume, overlap, label_map);
    }

    return ;
}


int relabel(seg_volume &test_volume, // test the labels of this volume for replacement
             volume_shape &block_offset,  // current block parameters
             vigra::detail::UnionFindArray<label_type> &label_map, // labels to be replaced
             std::map<label_type,three_set> &supervoxels // coordinate lists for each label
             ){
    int max = 0;
    // replace wrong labels
    iter3_t i3_f = test_volume.traverser_begin();
    iter3_t i3_l = test_volume.traverser_end();
    int i = 0;
    // iterate over the third dimension
    for ( ; i3_f != i3_l; ++i3_f, i++){
        iter2_t i2_f = i3_f.begin ();
        iter2_t i2_l = i3_f.end ();
        int j = 0;
        // iterate over the second dimension
        for ( ; i2_f != i2_l; ++i2_f, j++){
            iter1_t i1_f = i2_f.begin ();
            iter1_t i1_l = i2_f.end ();
            int k = 0;
            // iterate over the first dimension
            for (; i1_f != i1_l; ++i1_f, k++){
                // replace old label (except background)
                label_type test_label = *i1_f;
                if(test_label != 0){
                    *i1_f = label_map[test_label];
                    three_set::value_type pos (block_offset[2]+i,block_offset[1]+j,block_offset[0]+k);
                    supervoxels[label_map[test_label]].push_back(pos);
                    if(label_map[test_label] > max){
                        max = label_map[test_label];
                    }
                }
            }
        }
    }

    return max;
}



int segmentationLabeling(FileConfiguration fc, TaskConfiguration tc){
    if(tc.Verbose){
        std::cout << "- Labeling segmentation data.\n";
    }

    vigra::HDF5File seg_file (tc.Filename,vigra::HDF5File::Open);
    seg_file.cd(PATH_SEG);

    // get volume size
    hssize_t num_dimensions = seg_file.getDatasetDimensions(NAME_SEG);
    vigra::ArrayVector<hsize_t> dim_shape = seg_file.getDatasetShape(NAME_SEG);
    if(tc.Verbose){
      std::cout << "Shape: (" << dim_shape[0] << "," << dim_shape[1]<< "," << dim_shape[2] << ")" << std::endl;
    }

    //hssize_t num_dimensions = seg_file.getDatasetDimensions(fc.SegmentationPath);
    //vigra::ArrayVector<hsize_t> dim_shape = seg_file.getDatasetShape(fc.SegmentationPath);
    //seg_file.get_dataset_size("volume",num_dimensions,dim_shape);

    // preconditions
    if(num_dimensions != 3){
        std::cerr << "segmentationLabeling: Input data has wrong dimensionality! Supplied: " << num_dimensions << " Required: 3\n";
    }

    // create new dataset for labels
    volume_shape dim_size (dim_shape[0],dim_shape[1],dim_shape[2]);
    seg_file.createDataset<3,seg_type>(fc.LabelsPath,dim_size,0, volume_shape(32,32,32), 2);


    // blocks counter
    int blocks = 0;
    int max_blocks = std::ceil(double(dim_shape[0])/double(tc.BlockSize[0])) *
                     std::ceil(double(dim_shape[1])/double(tc.BlockSize[1])) *
                     std::ceil(double(dim_shape[2])/double(tc.BlockSize[2]));

    // label counter
    label_type max_label = 0;

    // Union-Find structure for label disambiguation
    vigra::detail::UnionFindArray<label_type> label_map(0);

    // go blockwise through volume
    for(coordinate_type o0 = 0; o0 < dim_shape[0]; o0 += tc.BlockSize[0]){
        for(coordinate_type o1 = 0; o1 < dim_shape[1]; o1 += tc.BlockSize[1]){
            for(coordinate_type o2 = 0; o2 < dim_shape[2]; o2 += tc.BlockSize[2]){

                // get the block size of current block.
                coordinate_type bs0 = (dim_shape[0]-o0 > tc.BlockSize[0]) ? tc.BlockSize[0] : (dim_shape[0]-o0);
                coordinate_type bs1 = (dim_shape[1]-o1 > tc.BlockSize[1]) ? tc.BlockSize[1] : (dim_shape[1]-o1);
                coordinate_type bs2 = (dim_shape[2]-o2 > tc.BlockSize[2]) ? tc.BlockSize[2] : (dim_shape[2]-o2);

                blocks ++;
                if(tc.Verbose){
                    std::cout << "  Searching block " << blocks << " of " << max_blocks << " with size ("<<bs0<<","<<bs1<<","<<bs2<<"). ";
                }

                // check for possible overlap with already labeled blocks
                coordinate_type ol0 = (o0 == 0) ? 0 : 1;
                coordinate_type ol1 = (o1 == 0) ? 0 : 1;
                coordinate_type ol2 = (o2 == 0) ? 0 : 1;

                // set offset and shape information
                volume_shape block_shape ( bs0, bs1, bs2 );
                volume_shape block_offset ( o0, o1, o2 );

                // load the selected block
                seg_volume src_vol (block_shape);
                seg_volume dest_vol (block_shape);
                seg_file.readBlock(NAME_SEG, block_offset, block_shape, src_vol);

                // do connected component analysis
                label_type newlabels = vigra::labelVolumeWithBackground(vigra::srcMultiArrayRange(src_vol), vigra::destMultiArray(dest_vol), vigra::NeighborCode3DSix(),0);

                if(tc.Verbose){
                    std::cout << newlabels << " new labels found.\n";
                }

                // shift labels by max_label
                //shiftLabels(dest_vol, max_label);
                vigra::transformMultiArray(vigra::srcMultiArrayRange(dest_vol),
                                           vigra::destMultiArray(dest_vol),
                                           vigra::functor::ifThenElse(
                                                   vigra::functor::Arg1() == vigra::functor::Param(0),
                                                   vigra::functor::Param(0),
                                                   vigra::functor::Arg1()+vigra::functor::Param(max_label)
                                                   )
                                           );


                max_label += newlabels;

                // create new labels
                for(int i = 0; i < newlabels; i++){
                    label_map.makeNewLabel();
                }

                // check for ambiguous labels at the borders
                checkBorders(fc.LabelsPath,dest_vol, seg_file, ol0, ol1, ol2,  block_offset, block_shape, label_map);

                seg_file.writeBlock(fc.LabelsPath, block_offset, dest_vol);

            }
        }
    }
//    std::cout << "Max label: " << max_label << "\n";
//    for(int i = 0; i < max_label+1; i++){
//        std::cout << label_map[i] << " ";
//    }
//    std::cout << "\n";

    label_map.makeNewLabel();
    label_map.makeContiguous();

//    std::cout << "Maxlabel: " << max_label <<"\n";
//    std::cout << "makeContiguous\n";
//    for(int i = 0; i < max_label+1; i++){
//        std::cout << label_map[i] << " ";
//    }
//    std::cout << "\n";

    // supervoxel groups
    std::map<label_type,three_set> supervoxels;

    // get new max_label
    max_label = 0;

    blocks = 0;
    // go blockwise through volume and give correct labels
    for(coordinate_type o0 = 0; o0 < dim_shape[0]; o0 += tc.BlockSize[0]){
        for(coordinate_type o1 = 0; o1 < dim_shape[1]; o1 += tc.BlockSize[1]){
            for(coordinate_type o2 = 0; o2 < dim_shape[2]; o2 += tc.BlockSize[2]){

                // get the block size of current block.
                coordinate_type bs0 = (dim_shape[0]-o0 > tc.BlockSize[0]) ? tc.BlockSize[0] : (dim_shape[0]-o0);
                coordinate_type bs1 = (dim_shape[1]-o1 > tc.BlockSize[1]) ? tc.BlockSize[1] : (dim_shape[1]-o1);
                coordinate_type bs2 = (dim_shape[2]-o2 > tc.BlockSize[2]) ? tc.BlockSize[2] : (dim_shape[2]-o2);

                blocks ++;
                if(tc.Verbose){
                    std::cout << "  Relabel block " << blocks << " of " << max_blocks << " with size ("<<bs0<<","<<bs1<<","<<bs2<<"). \n";
                }

                // set offset and shape information
                volume_shape block_shape ( bs0, bs1, bs2 );
                volume_shape block_offset ( o0, o1, o2 );

                // load the selected block
                seg_volume src_vol (block_shape);
                seg_file.readBlock(fc.LabelsPath, block_offset, block_shape, src_vol);

                // give new, unified labels
                int m = relabel(src_vol, block_offset, label_map, supervoxels);
                if( m > max_label){
                    max_label = m;
                }

                seg_file.writeBlock(fc.LabelsPath, block_offset, src_vol);

            }
        }
    }

    // write out geometry information

    // set the max-labels entry
    label_array max_labels (array_shape(4));
    max_labels[3] = max_label;

    // set up parts counters
    counter_type three_counter ( array_shape(max_label), ulong (0));
    for(label_type i = 0; i < max_label; i++){
        if(supervoxels.find(i+1) != supervoxels.end()){
            three_counter[i] = 1;
        }
    }

    counter_type other_counter (array_shape(1));

    // set up segmentation shape
    dim_size_type segmentation_shape (array_shape(3));
    segmentation_shape(0) = dim_shape[0];
    segmentation_shape(1) = dim_shape[1];
    segmentation_shape(2) = dim_shape[2];

    seg_file.cd_mk("/geometry");

    // write everything to file
    seg_file.write("max-labels",max_labels);

    seg_file.write("parts-counters-1",other_counter);

    seg_file.write("parts-counters-2",other_counter);

    seg_file.write("parts-counters-3",three_counter);

    seg_file.write("segmentation-shape",segmentation_shape);

    seg_file.mkdir("3-sets");
    seg_file.cd("3-sets");

    if(tc.Verbose){
        std::cout << "- Storing labels.\n";
    }
    for(int i = 1; i <= max_label; i++){
        std::map<label_type,three_set>::iterator it = supervoxels.find(i);
        if(it != supervoxels.end() ){
            char dir [20];
            sprintf(dir,"bin-%i",it->first);
            seg_file.cd_mk(dir);
            int size = it->second.size();
            set_type set (set_shape(size,3));
            for(int j = 0; j < 3; j++){
                for(int k = 0; k < size; k++){
                    set(k,j) = 2*(it->second[k][2-j]);
                }
            }
            char name [20];
            sprintf(name,"%i-1",it->first);

            seg_file.write(name,set);

            seg_file.cd_up();
        }
    }

    if(tc.Verbose){
        std::cout << "  Found " << max_label << " labels.\n";
    }

    return max_label;
}








// Label a complete volume at once
int segmentationLabelingAtOnce(FileConfiguration fc, TaskConfiguration tc){
    if(tc.Verbose){
        std::cout << "-- Labeling segmentation data.\n";
    }

    vigra::HDF5File seg_file (tc.Filename,vigra::HDF5File::Open);
    seg_file.cd(PATH_SEG);

    // get volume size
    hssize_t num_dimensions = seg_file.getDatasetDimensions(NAME_SEG);
    vigra::ArrayVector<hsize_t> dim_shape = seg_file.getDatasetShape(NAME_SEG);

    // preconditions
    if(num_dimensions != 3){
        std::cerr << "segmentationLabeling: Input data has wrong dimensionality! Supplied: " << num_dimensions << " Required: 3\n";
    }


    // create volumes and load data
    // ----------------------------
    volume_shape dim_size (dim_shape[0],dim_shape[1],dim_shape[2]);

    label_volume binary (dim_size);
    label_volume labels (dim_size);

    seg_file.read(NAME_SEG,binary);


    // do connected component analysis
    // -------------------------------
    label_type max_label = vigra::labelVolumeWithBackground(vigra::srcMultiArrayRange(binary),
                                                            vigra::destMultiArray(labels),
                                                            vigra::NeighborCode3DSix(),0);

    if(tc.Verbose){
        std::cout << "  " << max_label << " labels found.\n";
    }


    // collect coordinates for each label
    // ----------------------------------
    std::map<label_type,three_set> supervoxels;


    iter3_t i3_f = labels.traverser_begin();
    iter3_t i3_l = labels.traverser_end();
    int i = 0;
    // iterate over the third dimension
    for ( ; i3_f != i3_l; ++i3_f, i++)
    {
        iter2_t i2_f = i3_f.begin ();
        iter2_t i2_l = i3_f.end ();

        int j = 0;
        // iterate over the second dimension
        for ( ; i2_f != i2_l; ++i2_f, j++)
        {
            iter1_t i1_f = i2_f.begin ();
            iter1_t i1_l = i2_f.end ();

            int k = 0;
            // iterate over the first dimension
            for (; i1_f != i1_l; ++i1_f, k++)
            {
                // add coordinates to the corresponding group
                label_type label = *i1_f;
                if(label != 0)
                {
                    three_set::value_type pos (i,j,k);
                    supervoxels[label].push_back(pos);
                }
            }
        }
    }


    // write out geometry information
    // ------------------------------

    // set the max-labels entry
    label_array max_labels (array_shape(4));
    max_labels[3] = max_label;

    // set up parts counters
    counter_type three_counter ( array_shape(max_label), ulong (0));
    for(label_type i = 0; i < max_label; i++)
    {
        if(supervoxels.find(i+1) != supervoxels.end())
        {
            three_counter[i] = 1;
        }
    }

    counter_type other_counter (array_shape(1));

    // set up segmentation shape
    dim_size_type segmentation_shape (array_shape(3));
    segmentation_shape(0) = dim_shape[0];
    segmentation_shape(1) = dim_shape[1];
    segmentation_shape(2) = dim_shape[2];

    seg_file.cd_mk("/geometry");

    // write everything to file
    seg_file.write("max-labels",max_labels);

    seg_file.write("parts-counters-1",other_counter);

    seg_file.write("parts-counters-2",other_counter);

    seg_file.write("parts-counters-3",three_counter);

    seg_file.write("segmentation-shape",segmentation_shape);

    seg_file.mkdir("3-sets");
    seg_file.cd("3-sets");

    if(tc.Verbose){
        std::cout << "- Storing labels.\n";
    }
    for(int i = 1; i <= max_label; i++){
        std::map<label_type,three_set>::iterator it = supervoxels.find(i);
        if(it != supervoxels.end() ){
            char dir [20];
            sprintf(dir,"bin-%i",it->first);
            seg_file.cd_mk(dir);
            int size = it->second.size();
            set_type set (set_shape(size,3));
            for(int j = 0; j < 3; j++){
                for(int k = 0; k < size; k++){
                    set(k,j) = 2*(it->second[k][2-j]);
                }
            }
            char name [20];
            sprintf(name,"%i-1",it->first);

            seg_file.write(name,set);

            seg_file.cd_up();
        }
    }

    if(tc.Verbose){
        std::cout << "  Labeling finished.\n";
    }

    return max_label;
}

