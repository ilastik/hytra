#include <iostream>
#include <string>
#include <math.h>

#include <vigra/hdf5impex.hxx>
#include <vigra/multi_array.hxx>
#include "vigra/labelvolume.hxx"
#include "vigra/union_find.hxx"

#define BOOST_TEST_MODULE TestSegmentationLabeling
#include <boost/test/unit_test.hpp>


#include <ConfigFeatures.hxx>
#include <segmentationLabeling.hxx>


BOOST_AUTO_TEST_SUITE(SegmentationLabeling)

BOOST_AUTO_TEST_CASE( ShiftLabels )
{
    // create a labeled volume
    seg_volume vol (volume_shape(100,8,12));
    for(int i = 0; i < 100; i++){
        for(int j = 0; j < 8; j++){
            for(int k = 0; k < 12; k++){
                vol(i,j,k) = i;
            }
        }
    }

    // shift labels by maxLabel
    int maxLabel = 42;
    shiftLabels(vol,maxLabel);

    // test if labels are shifted correctly
    for(int i = 0; i < 100; i++){
        for(int j = 0; j < 8; j++){
            for(int k = 0; k < 12; k++){
                if(vol(i,j,k) != 0){
                    BOOST_REQUIRE(vol(i,j,k) == (i + maxLabel));
                }
            }
        }
    }
}


BOOST_AUTO_TEST_CASE( FindUnions )
{
    // create a weird 3D object with different labels
    seg_volume vol (volume_shape(100,80,120));
    for(int i = 0; i < 100; i++){
        for(int j = 0; j < 80; j++){
            for(int k = 0; k < 120; k++){
                double dist1 = std::sqrt(std::pow((i-50),2)+std::pow((j-40),2)+std::pow((k-60),2));
                double dist2 = std::sqrt(std::pow((i-50),2)+std::pow((j-40),2));
                double dist3 = std::sqrt(std::pow((i-50),2)+std::pow((k-60),2));
                double dist4 = std::sqrt(std::pow((k-60),2)+std::pow((j-40),2));
                if(dist1 < 20 && dist1 > 10 && dist2>3 && dist3>3 && dist4>3){
                    int label = 1;
                    if(i>50) label += 1;
                    if(j>40) label += 2;
                    if(k>60) label += 4;
                    vol(i,j,k) = label;
                }else{
                    vol(i,j,k) = 0;
                }

            }
        }
    }

    vigra::HDF5File f ("testSegmentationLabeling.h5",vigra::HDF5File::New);
    f.cd_mk("/segmentation/");
    f.write("volume",vol);
    f.flushToDisk();

    // prepare block with overlap
    volume_shape block_shape ( 50, 40, 60 );
    volume_shape block_offset ( 50, 40, 60 );

    seg_volume current (block_shape);
    f.readBlock("volume", block_offset, block_shape, current);

    // whole object labeled with '8'
    for(int i = 0; i < block_shape[0]; i++){
        for(int j = 0; j < block_shape[1]; j++){
            for(int k = 0; k < block_shape[2]; k++){
                if(current(i,j,k) != 0)
                    current(i,j,k) = 8;
            }
        }
    }

    // prepare slices for comparison
    volume_shape slice_shape1 ( 1, 40, 60 );
    volume_shape slice_offset1 ( 50-1, 40, 60 );
    volume_shape slice_shape2 ( 50, 1, 60 );
    volume_shape slice_offset2 ( 50, 40-1, 60 );
    volume_shape slice_shape3 ( 50, 40, 1 );
    volume_shape slice_offset3 ( 50, 40, 60-1 );


    seg_volume slice1 (slice_shape1);
    f.readBlock("volume", slice_offset1, slice_shape1, slice1);
    seg_volume slice2 (slice_shape2);
    f.readBlock("volume", slice_offset2, slice_shape2, slice2);
    seg_volume slice3 (slice_shape3);
    f.readBlock("volume", slice_offset3, slice_shape3, slice3);

    // prepare label map
    vigra::detail::UnionFindArray<label_type> label_map(0);
    for(int i = 0; i < 8; i++){
        label_map.makeNewLabel();
    }

    // run function
    findUnions(current,slice1,label_map);
    findUnions(current,slice2,label_map);
    findUnions(current,slice3,label_map);


    // unify labels
    label_map.makeContiguous();

    BOOST_CHECK(label_map[0] == 0);
    BOOST_CHECK(label_map[1] == 1);
    BOOST_CHECK(label_map[2] == 2);
    BOOST_CHECK(label_map[3] == 3);
    BOOST_CHECK(label_map[4] == 4);
    BOOST_CHECK(label_map[5] == 5);

    // labels 6,7,8 are connected to 4
    BOOST_CHECK(label_map[6] == 4);
    BOOST_CHECK(label_map[7] == 4);
    BOOST_CHECK(label_map[8] == 4);

}

BOOST_AUTO_TEST_CASE( CheckBorders )
{
    // create a weird 3D object with different labels
    seg_volume vol (volume_shape(100,80,120));
    for(int i = 0; i < 100; i++){
        for(int j = 0; j < 80; j++){
            for(int k = 0; k < 120; k++){
                double dist1 = std::sqrt(std::pow((i-50),2)+std::pow((j-40),2)+std::pow((k-60),2));
                double dist2 = std::sqrt(std::pow((i-50),2)+std::pow((j-40),2));
                double dist3 = std::sqrt(std::pow((i-50),2)+std::pow((k-60),2));
                double dist4 = std::sqrt(std::pow((k-60),2)+std::pow((j-40),2));
                if(dist1 < 20 && dist1 > 10 && dist2>3 && dist3>3 && dist4>3){
                    int label = 1;
                    if(i>50) label += 1;
                    if(j>40) label += 2;
                    if(k>60) label += 4;
                    vol(i,j,k) = label;
                }else{
                    vol(i,j,k) = 0;
                }

            }
        }
    }

    vigra::HDF5File f ("testSegmentationLabeling.h5",vigra::HDF5File::New);
    f.cd_mk("/segmentation/");
    f.write("labels",vol);
    f.flushToDisk();

    // prepare block with overlap
    volume_shape block_shape ( 50, 40, 60 );
    volume_shape block_offset ( 50, 40, 60 );

    seg_volume current (block_shape);
    f.readBlock("labels", block_offset, block_shape, current);

    // whole object labeled with '8'
    for(int i = 0; i < block_shape[0]; i++){
        for(int j = 0; j < block_shape[1]; j++){
            for(int k = 0; k < block_shape[2]; k++){
                if(current(i,j,k) != 0)
                    current(i,j,k) = 8;
            }
        }
    }

    // prepare label map
    vigra::detail::UnionFindArray<label_type> label_map(0);
    for(int i = 0; i < 8; i++){
        label_map.makeNewLabel();
    }


    checkBorders(current,f, 1,1,1, block_shape, block_offset, label_map);

    // unify labels
    label_map.makeContiguous();


    BOOST_CHECK(label_map[0] == 0);
    BOOST_CHECK(label_map[1] == 1);
    BOOST_CHECK(label_map[2] == 2);
    BOOST_CHECK(label_map[3] == 3);
    BOOST_CHECK(label_map[4] == 4);
    BOOST_CHECK(label_map[5] == 5);

    // labels 6,7,8 are connected to 4
    BOOST_CHECK(label_map[6] == 4);
    BOOST_CHECK(label_map[7] == 4);
    BOOST_CHECK(label_map[8] == 4);

}



BOOST_AUTO_TEST_CASE( Relabel )
{
    // create a weird 3D object with different labels
    seg_volume vol (volume_shape(100,80,120));
    for(int i = 0; i < 100; i++){
        for(int j = 0; j < 80; j++){
            for(int k = 0; k < 120; k++){
                double dist1 = std::sqrt(std::pow((i-50),2)+std::pow((j-40),2)+std::pow((k-60),2));
                double dist2 = std::sqrt(std::pow((i-50),2)+std::pow((j-40),2));
                double dist3 = std::sqrt(std::pow((i-50),2)+std::pow((k-60),2));
                double dist4 = std::sqrt(std::pow((k-60),2)+std::pow((j-40),2));
                if(dist1 < 20 && dist1 > 10 && dist2>3 && dist3>3 && dist4>3){
                    int label = 1;
                    if(i>50) label += 1;
                    if(j>40) label += 2;
                    if(k>60) label += 4;
                    vol(i,j,k) = label;
                }else{
                    vol(i,j,k) = 0;
                }

            }
        }
    }

    vigra::HDF5File f ("testSegmentationLabeling.h5",vigra::HDF5File::New);
    f.cd_mk("/segmentation/");
    f.write("labels",vol);
    f.flushToDisk();

    // prepare block with overlap
    volume_shape block_shape ( 50, 40, 60 );
    volume_shape block_offset ( 50, 40, 60 );

    seg_volume current (block_shape);
    f.readBlock("labels", block_offset, block_shape, current);

    // whole object labeled with '8'
    for(int i = 0; i < block_shape[0]; i++){
        for(int j = 0; j < block_shape[1]; j++){
            for(int k = 0; k < block_shape[2]; k++){
                if(current(i,j,k) != 0)
                    current(i,j,k) = 8;
            }
        }
    }

    // prepare label map
    vigra::detail::UnionFindArray<label_type> label_map(0);
    for(int i = 0; i < 8; i++){
        label_map.makeNewLabel();
    }


    checkBorders(current,f, 1,1,1, block_shape, block_offset, label_map);
    label_map.makeContiguous();

    // supervoxel groups
    std::map<label_type,three_set> supervoxels;

    // relabel the volume and collect supervoxel groups
    // go blockwise through volume and give correct labels
    for(int o0 = 0; o0 < 100; o0 += 50){
        for(int o1 = 0; o1 < 80; o1 += 40){
            for(int o2 = 0; o2 < 120; o2 += 60){

                // set offset and shape information
                volume_shape block_shape ( 50, 40, 60 );
                volume_shape block_offset ( o0, o1, o2 );

                // load the selected block
                seg_volume src_vol (block_shape);
                f.readBlock("labels", block_offset, block_shape, src_vol);

                // give new, unified labels
                int m = relabel(src_vol, block_offset, label_map, supervoxels);

                f.writeBlock("labels", block_offset, src_vol);

            }
        }
    }

    BOOST_CHECK( (supervoxels.find(1) != supervoxels.end()) );
    BOOST_CHECK( (supervoxels.find(2) != supervoxels.end()) );
    BOOST_CHECK( (supervoxels.find(3) != supervoxels.end()) );
    BOOST_CHECK( (supervoxels.find(4) != supervoxels.end()) );
    BOOST_CHECK( (supervoxels.find(5) != supervoxels.end()) );
    BOOST_CHECK( (supervoxels.find(6) == supervoxels.end()) );
    BOOST_CHECK( (supervoxels.find(7) == supervoxels.end()) );
    BOOST_CHECK( (supervoxels.find(8) == supervoxels.end()) );

    f.read("labels",vol);
    for(int i = 0; i < 100; i++){
        for(int j = 0; j < 80; j++){
            for(int k = 0; k < 120; k++){
                if(vol(i,j,k) != 0){
                    int label = 1;
                    if(i>50) label += 1;
                    if(j>40) label += 2;
                    if(k>60) label += 4;
                    BOOST_CHECK( vol(i,j,k) == label_map[label] );
                }

            }
        }
    }

}


BOOST_AUTO_TEST_CASE( SegmentationLabeling )
{
    {
        // create a weird 3D segmentation
        seg_volume vol (volume_shape(100,80,120));
        for(int i = 0; i < 100; i++){
            for(int j = 0; j < 80; j++){
                for(int k = 0; k < 120; k++){
                    double dist1 = std::sqrt(std::pow((i-50),2)+std::pow((j-40),2)+std::pow((k-60),2));
                    double dist2 = std::sqrt(std::pow((i-50),2)+std::pow((j-40),2));
                    double dist3 = std::sqrt(std::pow((i-50),2)+std::pow((k-60),2));
                    double dist4 = std::sqrt(std::pow((k-60),2)+std::pow((j-40),2));
                    if(dist1 < 20 && dist1 > 10 && dist2>3 && dist3>3 && dist4>3 ){
                        vol(i,j,k) = 1;
                    }else{
                        vol(i,j,k) = 0;
                    }

                }
            }
        }

        vigra::HDF5File f ("testSegmentationLabeling.h5",vigra::HDF5File::New);
        f.cd_mk("/segmentation/");
        f.write("volume",vol);
        f.flushToDisk();
    }

    segmentationLabeling("testSegmentationLabeling.h5",volume_shape(50,40,60));

    seg_volume vol (volume_shape(100,80,120));
    vigra::HDF5File f ("testSegmentationLabeling.h5",vigra::HDF5File::Open);
    f.read("/segmentation/labels",vol);
    for(int i = 0; i < 100; i++){
        for(int j = 0; j < 80; j++){
            for(int k = 0; k < 120; k++){
                if(vol(i,j,k) != 0){
                    BOOST_REQUIRE( vol(i,j,k) == 1 );
                }
            }
        }
    }
}


BOOST_AUTO_TEST_CASE( SegmentationLabeling2 )
{
    {
        // create some cells
        seg_volume vol (volume_shape(100,100,100));
        for(int i = 0; i < 100; i++){
            for(int j = 0; j < 100; j++){
                for(int k = 0; k < 100; k++){
                    vol(i,j,k) = 0;
                    for(int a = 10; a < 100; a += 40){
                        for(int b = 10; b < 100; b += 40){
                            for(int c = 10; c < 100; c += 40){
                               double dist = std::sqrt(std::pow((i-a),2)+std::pow((j-b),2)+std::pow((k-c),2));
                               if(dist < 5){
                                   vol(i,j,k) = 1;
                               }
                            }
                        }
                    }

                }
            }
        }

        vigra::HDF5File f ("testSegmentationLabeling2.h5",vigra::HDF5File::New);
        f.cd_mk("/segmentation/");
        f.write("volume",vol);
        f.flushToDisk();
    }

    segmentationLabeling("testSegmentationLabeling2.h5",volume_shape(50,50,50));


    // this test was created for manual inspection. You can automate it if you want.
}

BOOST_AUTO_TEST_SUITE_END()
