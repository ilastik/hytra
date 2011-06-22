#include <iostream>
#include <string>
#include <math.h>

#include <vigra/hdf5impex.hxx>
#include <vigra/multi_array.hxx>
#include "vigra/labelvolume.hxx"
#include "vigra/union_find.hxx"

#define BOOST_TEST_MODULE TestReadSuperVoxels
#include <boost/test/unit_test.hpp>


#include <ConfigFeatures.hxx>
#include <segmentationLabeling.hxx>
#include <readSuperVoxels.hxx>

BOOST_AUTO_TEST_SUITE(ReadSuperVoxels)

BOOST_AUTO_TEST_CASE( readSuperVoxels )
{

    // create some artificial cell nuclei
    volume_shape pos1 (81,45,22);
    volume_shape pos2 (90,66,100);
    volume_shape pos3 (35,8,105);
    volume_shape pos4 (50,71,34);
    volume_shape pos5 (10,28,85);
    seg_volume vol (volume_shape(100,80,120));
    for(int i = 0; i < 100; i++){
        for(int j = 0; j < 80; j++){
            for(int k = 0; k < 120; k++){
                double dist1 = std::sqrt(std::pow((i-pos1[0]),2)+std::pow((j-pos1[1]),2)+std::pow((k-pos1[2]),2));
                double dist2 = std::sqrt(std::pow((i-pos2[0]),2)+std::pow((j-pos2[1]),2)+std::pow((k-pos2[2]),2));
                double dist3 = std::sqrt(std::pow((i-pos3[0]),2)+std::pow((j-pos3[1]),2)+std::pow((k-pos3[2]),2));
                double dist4 = std::sqrt(std::pow((i-pos4[0]),2)+std::pow((j-pos4[1]),2)+std::pow((k-pos4[2]),2));
                double dist5 = std::sqrt(std::pow((i-pos5[0]),2)+std::pow((j-pos5[1]),2)+std::pow((k-pos5[2]),2));
                if(dist1 < 5 || dist2 < 5 || dist3 < 5 || dist4 < 5 || dist5 < 5 ){
                    vol(i,j,k) = 1;
                }else{
                    vol(i,j,k) = 0;
                }

            }
        }
    }

    {
        vigra::HDF5File f ("testSuperVoxels.h5",vigra::HDF5File::New);
        f.cd_mk("/segmentation/");
        f.write("volume",vol);
        f.cd_mk("/raw/");
        f.write("volume",vol);
        f.flushToDisk();
    }

    segmentationLabeling("testSuperVoxels.h5", volume_shape(50,40,60));

    dSuperVoxelReader r ("testSuperVoxels.h5");

    BOOST_REQUIRE(r.maxLabel(3) == 5);

    label_array la = r.labelContent();
    for(int i = 0; i < r.maxLabel(3); i++){
        BOOST_CHECK(la[i] == 1);
    }

    three_set l1 = r.threeSet(1);
    three_set l2 = r.threeSet(2);
    three_set l3 = r.threeSet(3);
    three_set l4 = r.threeSet(4);
    three_set l5 = r.threeSet(5);

    BOOST_CHECK(l1.size() == 485);
    BOOST_CHECK(l2.size() == 485);
    BOOST_CHECK(l3.size() == 485);
    BOOST_CHECK(l4.size() == 485);
    BOOST_CHECK(l5.size() == 485);

    for(int i = 0; i < 485; i++){
        double dist5 = std::sqrt(std::pow((l1[i][0]/2-pos5[0]),2)+std::pow((l1[i][1]/2-pos5[1]),2)+std::pow((l1[i][2]/2-pos5[2]),2));
        double dist3 = std::sqrt(std::pow((l2[i][0]/2-pos3[0]),2)+std::pow((l2[i][1]/2-pos3[1]),2)+std::pow((l2[i][2]/2-pos3[2]),2));
        double dist4 = std::sqrt(std::pow((l3[i][0]/2-pos4[0]),2)+std::pow((l3[i][1]/2-pos4[1]),2)+std::pow((l3[i][2]/2-pos4[2]),2));
        double dist1 = std::sqrt(std::pow((l4[i][0]/2-pos1[0]),2)+std::pow((l4[i][1]/2-pos1[1]),2)+std::pow((l4[i][2]/2-pos1[2]),2));
        double dist2 = std::sqrt(std::pow((l5[i][0]/2-pos2[0]),2)+std::pow((l5[i][1]/2-pos2[1]),2)+std::pow((l5[i][2]/2-pos2[2]),2));
        BOOST_CHECK(dist5 < 5);
        BOOST_CHECK(dist3 < 5);
        BOOST_CHECK(dist4 < 5);
        BOOST_CHECK(dist1 < 5);
        BOOST_CHECK(dist2 < 5);
    }

}

BOOST_AUTO_TEST_SUITE_END()

