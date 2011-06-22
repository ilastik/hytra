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
#include <dFeatureExtractor.hxx>

BOOST_AUTO_TEST_SUITE(FeatureExtractorTestSuite)

BOOST_AUTO_TEST_CASE( testFeatureExtractor )
{

    // create some artificial cell nuclei
    volume_shape pos1 (81,45,22);
    volume_shape pos2 (90,66,100);
    volume_shape pos3 (35,8,105);
    volume_shape pos4 (50,71,34);
    volume_shape pos5 (10,28,85);
    seg_volume seg (volume_shape(100,80,120));
    seg_volume raw (volume_shape(100,80,120));
    for(int i = 0; i < 100; i++){
        for(int j = 0; j < 80; j++){
            for(int k = 0; k < 120; k++){
                double dist1 = std::sqrt(std::pow((i-pos1[0]),2)+std::pow((j-pos1[1]),2)+std::pow((k-pos1[2]),2));
                double dist2 = std::sqrt(std::pow((i-pos2[0]),2)+std::pow((j-pos2[1]),2)+std::pow((k-pos2[2]),2));
                double dist3 = std::sqrt(std::pow((i-pos3[0]),2)+std::pow((j-pos3[1]),2)+std::pow((k-pos3[2]),2));
                double dist4 = std::sqrt(std::pow((i-pos4[0]),2)+std::pow((j-pos4[1]),2)+std::pow((k-pos4[2]),2));
                double dist5 = std::sqrt(std::pow((i-pos5[0]),2)+std::pow((j-pos5[1]),2)+std::pow((k-pos5[2]),2));
                if(dist1 < 5 || dist2 < 5 || dist3 < 5 || dist4 < 5 || dist5 < 5 ){
                    seg(i,j,k) = 1;
                    if(dist1 < 3 || dist2 < 3 || dist3 < 3 || dist4 < 3 || dist5 < 3 ){
                        raw(i,j,k) = 1500;
                    }else{
                        raw(i,j,k) = 1000;
                    }

                }else{
                    seg(i,j,k) = 0;
                    raw(i,j,k) = 0;
                }

            }
        }
    }

    {
        vigra::HDF5File f ("testFeatureExtractor.h5",vigra::HDF5File::New);
        f.cd_mk("/segmentation/");
        f.write("volume",seg);
        f.cd_mk("/raw/");
        f.write("volume",raw);
        f.flushToDisk();
    }

    FileConfiguration fc;
    TaskConfiguration tc ("testFeatureExtractor.h5",false);
    tc.BlockSize = volume_shape(50,40,60);
    segmentationLabeling(fc,tc);

    features::dFeatureExtractor fext ( "testFeatureExtractor.h5", features::FT_ALL, false);
    fext.extract_features();

    vigra::HDF5File f ("testFeatureExtractor.h5",vigra::HDF5File::Open);
    unsigned short a = 0;
    f.readAtomic("/features/supervoxels", a);

    // there should be 5 supervoxels
    BOOST_REQUIRE(a == 5);

    // each supervoxel should consist of 485 voxels
    double vol = 0;
    f.readAtomic("/features/1/volume", vol);
    BOOST_CHECK(vol == 485);

    f.readAtomic("/features/2/volume", vol);
    BOOST_CHECK(vol == 485);

    f.readAtomic("/features/3/volume", vol);
    BOOST_CHECK(vol == 485);

    f.readAtomic("/features/4/volume", vol);
    BOOST_CHECK(vol == 485);

    f.readAtomic("/features/5/volume", vol);
    BOOST_CHECK(vol == 485);

    // test bounding box
    for(int i = 1; i <= 5; i++){
        std::stringstream bbox_path;
        bbox_path << "/features/" << i << "/bbox";
        feature_array bbox (array_shape(f.getDatasetShape(bbox_path.str())[0]));
        f.read(bbox_path.str(),bbox);
        BOOST_CHECK(bbox[3]-bbox[0] == 8);
        BOOST_CHECK(bbox[4]-bbox[1] == 8);
        BOOST_CHECK(bbox[5]-bbox[2] == 8);
        BOOST_CHECK_CLOSE(bbox[6], 2./3.,1);
    }

    // test weighted and unweighted mean position
    for(int i = 1; i <= 5; i++){
        std::stringstream com_path;
        std::stringstream pos_path;
        com_path << "/features/" << i << "/com";
        pos_path << "/features/" << i << "/position";
        feature_array com (array_shape(f.getDatasetShape(com_path.str())[0]));
        feature_array pos (array_shape(f.getDatasetShape(pos_path.str())[0]));
        f.read(com_path.str(),com);
        f.read(pos_path.str(),pos);
        // cells are symmetric in our case
        BOOST_CHECK(com[0] == pos[0]);
        BOOST_CHECK(com[1] == pos[1]);
        BOOST_CHECK(com[2] == pos[2]);
        BOOST_CHECK(com[6] == 0);
        BOOST_CHECK(com[7] == 0);
        BOOST_CHECK(com[8] == 0);
        BOOST_CHECK(pos[6] == 0);
        BOOST_CHECK(pos[7] == 0);
        BOOST_CHECK(pos[8] == 0);

        // cells have higher intensity in the center
        //--> variance of weighted mean is lower
        BOOST_CHECK(com[3] <= pos[3]);
        BOOST_CHECK(com[4] <= pos[4]);
        BOOST_CHECK(com[5] <= pos[5]);

        //--> kurtosis of weighted mean is higher (more peak-shaped)
        BOOST_CHECK(com[9] >= pos[9]);
        BOOST_CHECK(com[10] >= pos[10]);
        BOOST_CHECK(com[11] >= pos[11]);
    }

    // test intensity
    for(int i = 1; i <= 5; i++){
        std::stringstream imm_path;
        std::stringstream int_path;
        imm_path << "/features/" << i << "/intminmax";
        int_path << "/features/" << i << "/intensity";
        feature_array imm (array_shape(f.getDatasetShape(imm_path.str())[0]));
        feature_array intensity (array_shape(f.getDatasetShape(int_path.str())[0]));
        f.read(imm_path.str(),imm);
        f.read(int_path.str(),intensity);
        // cells are symmetric in our case
        BOOST_CHECK(imm[0] == 1000);
        BOOST_CHECK(imm[1] == 1500);

        double mean = (93.*1500. + 392.*1000.)/485.;
        BOOST_CHECK_CLOSE(intensity[0], mean, 0.01);

        double var = (93.*pow(1500.-mean,2) + 392.*pow(1000.-mean,2))/485.;
        BOOST_CHECK_CLOSE(intensity[1], var, 0.01);

        double skew =(93.*pow(1500.-mean,3) + 392.*pow(1000.-mean,3))/485./pow(var,3./2.);
        BOOST_CHECK_CLOSE(intensity[2], skew, 0.01);

        double kurt =(93.*pow(1500.-mean,4) + 392.*pow(1000.-mean,4))/485./pow(var,2.) - 3;
        BOOST_CHECK_CLOSE(intensity[3], kurt, 0.01);
    }


    // test principal components
    for(int i = 1; i <= 5; i++){
        std::stringstream pc_path;
        pc_path << "/features/" << i << "/pc";
        feature_array pc (array_shape(f.getDatasetShape(pc_path.str())[0]));
        f.read(pc_path.str(),pc);
        // cells are symmetric in our case
        BOOST_CHECK(pc[0] == pc[1]);
        BOOST_CHECK(pc[1] == pc[2]);

        double len = sqrt(pow(pc[3],2)+pow(pc[4],2)+pow(pc[5],2));
        BOOST_CHECK_CLOSE(len, 1, 0.01);

        len = sqrt(pow(pc[6],2)+pow(pc[7],2)+pow(pc[8],2));
        BOOST_CHECK_CLOSE(len, 1, 0.01);

        len = sqrt(pow(pc[9],2)+pow(pc[10],2)+pow(pc[11],2));
        BOOST_CHECK_CLOSE(len, 1, 0.01);

        double lin_dep = pc[3]*pc[6] + pc[4]*pc[7] + pc[5]*pc[8];
        BOOST_CHECK_CLOSE(lin_dep, 0, 0.01);

        lin_dep = pc[3]*pc[9] + pc[4]*pc[10] + pc[5]*pc[11];
        BOOST_CHECK_CLOSE(lin_dep, 0, 0.01);

        lin_dep = pc[9]*pc[6] + pc[10]*pc[7] + pc[11]*pc[8];
        BOOST_CHECK_CLOSE(lin_dep, 0, 0.01);
    }



}

BOOST_AUTO_TEST_SUITE_END()

