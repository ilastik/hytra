#include "unittest.hxx"

#include "object_features/objectFeatures.hxx"


void testFeatureVolume () {
    three_set coordinates;
    value_set intensities;

    feature_array volume = features::extractVolume(coordinates,intensities);
    should(volume[0] == 0);

    coordinates.push_back( three_coordinate(1,1,1) );
    intensities.push_back(1);

    volume = features::extractVolume(coordinates,intensities);
    should(volume[0] == 1);

    for(int i = 0; i < 100000; i++){
        coordinates.push_back( three_coordinate(1,1,1) );
        intensities.push_back(1);
    }

    volume = features::extractVolume(coordinates,intensities);
    should(volume[0] == 100001);

    // test if feature function and feature class produce same results
    features::ObjectVolume<unsigned short,3> o ( coordinates,intensities);
    feature_array volume_obj = o.get();

    shouldEqual(volume_obj[0], volume[0]);

    // test the Volume class
    features::ObjectVolume<short,3>::coordinates_type coords;
    features::ObjectVolume<short,3>::values_type intens;

    for(int i = 0; i < 1500; i++){
        coords.push_back(vigra::MultiArrayShape<3>::type(i,i+1,i+2));
        intens.push_back(short(i));
    }

    features::ObjectVolume<short,3> v1 (coords, intens);

    for(int i = 0; i < 500; i++){
        coords.push_back(vigra::MultiArrayShape<3>::type(i,i+1,i+2));
        intens.push_back(short(i));
    }

    features::ObjectVolume<short,3> v2 (coords, intens);

    shouldEqual(v1.get()(0) , 1500);
    shouldEqual(v2.get()(0) , 2000);

    v1.mergeWith(v2);
    shouldEqual(v1.get()(0) , 3500);
}



void testFeatureBoundingBox () {
    three_set coordinates;
    value_set intensities;

    feature_array bbox = features::extractBoundingBox(coordinates,intensities);
    for(int i = 0; i < 7; i++)
        should(bbox(i) == -1);

    coordinates.push_back( three_coordinate(1,2,3) );
    intensities.push_back(1);

    bbox = features::extractBoundingBox(coordinates,intensities);
    should(bbox(0) == 1);
    should(bbox(1) == 2);
    should(bbox(2) == 3);
    should(bbox(3) == 1);
    should(bbox(4) == 2);
    should(bbox(5) == 3);
    should(bbox(6) == 1);

    for(int i = 0; i < 100000; i++){
        coordinates.push_back( three_coordinate(i+2,i+3,i+4) );
        intensities.push_back(1);
    }
    bbox = features::extractBoundingBox(coordinates,intensities);
    should(bbox(0) == 1);
    should(bbox(1) == 2);
    should(bbox(2) == 3);
    should(bbox(3) == 100001);
    should(bbox(4) == 100002);
    should(bbox(5) == 100003);
    shouldEqualTolerance(bbox(6), 100001./(100001.*100001.*100001.), 0.000001);


    // test if feature function and feature class produce same results
    features::ObjectBoundingBox<unsigned short,3> o ( coordinates,intensities);
    feature_array bbox_obj = o.get();

    shouldEqual(bbox_obj[0], bbox[0]);
    shouldEqual(bbox_obj[1], bbox[1]);
    shouldEqual(bbox_obj[2], bbox[2]);
    shouldEqual(bbox_obj[3], bbox[3]);
    shouldEqual(bbox_obj[4], bbox[4]);
    shouldEqual(bbox_obj[5], bbox[5]);
    shouldEqual(bbox_obj[6], bbox[6]);


    // more tests in higher dimensions

}



void testFeaturePosition () {
    three_set coordinates;
    value_set intensities;

    feature_array position = features::extractPosition(coordinates,intensities);
    for(int i = 0; i < 12; i++)
        should(position(i) == -1);

    coordinates.push_back( three_coordinate(1,1,1) );
    intensities.push_back(1);

    position = features::extractPosition(coordinates,intensities);
    should(position(0) == 1);
    should(position(1) == 1);
    should(position(2) == 1);
    for(int i = 3; i < 12; i++)
        should(position(i) == -1);

    coordinates.push_back( three_coordinate(2,1,1) ); intensities.push_back(1);
    coordinates.push_back( three_coordinate(3,1,1) ); intensities.push_back(1);
    coordinates.push_back( three_coordinate(4,1,1) ); intensities.push_back(1);
    coordinates.push_back( three_coordinate(5,1,1) ); intensities.push_back(1);


    position = features::extractPosition(coordinates,intensities);
    should(position(0) == 3);
    should(position(1) == 1);
    should(position(2) == 1);
    should(position(3) == 2.);
    should(position(4) == 0);
    should(position(5) == 0);
    should(position(6) == 0.);
    should(position(7) == -1);
    should(position(8) == -1);
    shouldEqualTolerance(position(9),34./4./5.-3., 0.000001);
    should(position(10)== -1);
    should(position(11)== -1);


    coordinates.push_back( three_coordinate(1,2,1) ); intensities.push_back(1);
    coordinates.push_back( three_coordinate(2,2,1) ); intensities.push_back(1);
    coordinates.push_back( three_coordinate(3,2,1) ); intensities.push_back(1);
    coordinates.push_back( three_coordinate(4,2,1) ); intensities.push_back(1);
    coordinates.push_back( three_coordinate(5,2,1) ); intensities.push_back(1);
    coordinates.push_back( three_coordinate(1,1,2) ); intensities.push_back(1);
    coordinates.push_back( three_coordinate(2,1,2) ); intensities.push_back(1);
    coordinates.push_back( three_coordinate(3,1,2) ); intensities.push_back(1);
    coordinates.push_back( three_coordinate(4,1,2) ); intensities.push_back(1);
    coordinates.push_back( three_coordinate(5,1,2) ); intensities.push_back(1);
    coordinates.push_back( three_coordinate(1,2,2) ); intensities.push_back(1);
    coordinates.push_back( three_coordinate(2,2,2) ); intensities.push_back(1);
    coordinates.push_back( three_coordinate(3,2,2) ); intensities.push_back(1);
    coordinates.push_back( three_coordinate(4,2,2) ); intensities.push_back(1);
    coordinates.push_back( three_coordinate(5,2,2) ); intensities.push_back(1);

    position = features::extractPosition(coordinates,intensities);
    should(position(0) == 3);
    should(position(1) == 1.5);
    should(position(2) == 1.5);
    should(position(3) == 2.);
    should(position(4) == 0.25);
    should(position(5) == 0.25);
    should(position(6) == 0);
    should(position(7) == 0);
    should(position(8) == 0);
    shouldEqualTolerance(position(9),34./4./5.-3., 0.000001);
    should(position(10)== 0.5*0.5*0.5*0.5*16.-3.);
    should(position(11)== 0.5*0.5*0.5*0.5*16.-3.);


    // test if feature function and feature class produce same results
    features::ObjectPosition<unsigned short,3> o ( coordinates,intensities);
    feature_array position_obj = o.get();

    shouldEqual(position_obj[0], position[0]);
    shouldEqual(position_obj[1], position[1]);
    shouldEqual(position_obj[2], position[2]);
    shouldEqual(position_obj[3], position[3]);
    shouldEqual(position_obj[4], position[4]);
    shouldEqual(position_obj[5], position[5]);
    shouldEqual(position_obj[6], position[6]);
    shouldEqual(position_obj[7], position[7]);
    shouldEqual(position_obj[8], position[8]);
    shouldEqual(position_obj[9]-3, position[9]);
    shouldEqual(position_obj[10]-3, position[10]);
    shouldEqual(position_obj[11]-3, position[11]);

    // test ObjectPosition class
    features::ObjectPosition<short,3>::coordinates_type coords1;
    features::ObjectPosition<short,3>::values_type intens1;

    features::ObjectPosition<short,3>::coordinates_type coords2;
    features::ObjectPosition<short,3>::values_type intens2;

    features::ObjectPosition<short,3>::coordinates_type coordsM;
    features::ObjectPosition<short,3>::values_type intensM;

    for(int i = 0; i < 5; i++){
        for(int j = 0; j < 5; j++){
            for(int k = 0; k < 5; k++){
                coords1.push_back(vigra::MultiArrayShape<3>::type(i,j,k));
                intens1.push_back(short(i));
                coords2.push_back(vigra::MultiArrayShape<3>::type(i+5,j,k));
                intens2.push_back(short(i));

                coordsM.push_back(vigra::MultiArrayShape<3>::type(i,j,k));
                intensM.push_back(short(i));
                coordsM.push_back(vigra::MultiArrayShape<3>::type(i+5,j,k));
                intensM.push_back(short(i));
            }
        }
    }

    features::ObjectPosition<short,3> p1 (coords1, intens1);
    features::ObjectPosition<short,3> p2 (coords2, intens2);
    features::ObjectPosition<short,3> pM (coordsM, intensM);

    p1.mergeWith(p2);
    for(int i = 0; i < 12; i++){
        shouldEqualTolerance(p1.get()[i],pM.get()[i], 0.000001);
    }
}



void testFeatureWeightedPosition () {
    three_set coordinates;
    value_set intensities;

    feature_array position = features::extractWeightedPosition(coordinates,intensities);
    for(int i = 0; i < 12; i++)
        should(position(i) == -1);

    coordinates.push_back( three_coordinate(1,1,1) );
    intensities.push_back(1);

    position = features::extractWeightedPosition(coordinates,intensities);
    should(position(0) == 1);
    should(position(1) == 1);
    should(position(2) == 1);
    for(int i = 3; i < 12; i++)
        should(position(i) == -1);

    coordinates.push_back( three_coordinate(2,1,1) ); intensities.push_back(3);
    coordinates.push_back( three_coordinate(3,1,1) ); intensities.push_back(5);
    coordinates.push_back( three_coordinate(4,1,1) ); intensities.push_back(3);
    coordinates.push_back( three_coordinate(5,1,1) ); intensities.push_back(1);

    position = features::extractWeightedPosition(coordinates,intensities);
    should(position(0) == 3);
    should(position(1) == 1);
    should(position(2) == 1);
    shouldEqualTolerance(position(3), 14./13., 0.000001);
    should(position(4) == 0);
    should(position(5) == 0);
    should(position(6) == 0);
    should(position(7) == -1);
    should(position(8) == -1);
    shouldEqualTolerance(position(9), 38./13./(14./13.*14./13.)-3., 0.000001);
    should(position(10)== -1);
    should(position(11)== -1);


    coordinates.push_back( three_coordinate(1,2,1) ); intensities.push_back(1);
    coordinates.push_back( three_coordinate(2,2,1) ); intensities.push_back(3);
    coordinates.push_back( three_coordinate(3,2,1) ); intensities.push_back(5);
    coordinates.push_back( three_coordinate(4,2,1) ); intensities.push_back(3);
    coordinates.push_back( three_coordinate(5,2,1) ); intensities.push_back(1);
    coordinates.push_back( three_coordinate(1,1,2) ); intensities.push_back(2);
    coordinates.push_back( three_coordinate(2,1,2) ); intensities.push_back(6);
    coordinates.push_back( three_coordinate(3,1,2) ); intensities.push_back(10);
    coordinates.push_back( three_coordinate(4,1,2) ); intensities.push_back(6);
    coordinates.push_back( three_coordinate(5,1,2) ); intensities.push_back(2);
    coordinates.push_back( three_coordinate(1,2,2) ); intensities.push_back(2);
    coordinates.push_back( three_coordinate(2,2,2) ); intensities.push_back(6);
    coordinates.push_back( three_coordinate(3,2,2) ); intensities.push_back(10);
    coordinates.push_back( three_coordinate(4,2,2) ); intensities.push_back(6);
    coordinates.push_back( three_coordinate(5,2,2) ); intensities.push_back(2);

    position = features::extractWeightedPosition(coordinates,intensities);
    should(position(0) == 3);
    should(position(1) == 1.5);
    shouldEqualTolerance(position(2), 5./3., 0.000001);
    shouldEqualTolerance(position(3), 14./13., 0.000001);
    shouldEqualTolerance(position(4), 0.25,  0.000001);
    shouldEqualTolerance(position(5), 2./9.,  0.000001);
    should(position(6) == 0);
    should(position(7) == 0);
    shouldEqualTolerance(position(8), -1./std::sqrt(2.), 0.000001);
    shouldEqualTolerance(position(9), 38./13./(14.*14/13./13.) -3., 0.000001);
    shouldEqualTolerance(position(10), 0.5*0.5*0.5*0.5*16. -3., 0.000001);
    shouldEqualTolerance(position(11), -1.5, 0.000001);


    // test if feature function and feature class produce same results
    features::ObjectWeightedPosition<unsigned short,3> o ( coordinates,intensities);
    feature_array position_obj = o.get();

    shouldEqual(position_obj[0], position[0]);
    shouldEqual(position_obj[1], position[1]);
    shouldEqual(position_obj[2], position[2]);
    shouldEqual(position_obj[3], position[3]);
    shouldEqual(position_obj[4], position[4]);
    shouldEqual(position_obj[5], position[5]);
    shouldEqual(position_obj[6], position[6]);
    shouldEqual(position_obj[7], position[7]);
    shouldEqual(position_obj[8], position[8]);
    shouldEqual(position_obj[9]-3, position[9]);
    shouldEqual(position_obj[10]-3, position[10]);
    shouldEqual(position_obj[11]-3, position[11]);


    // test ObjectWeightedPosition class
    features::ObjectWeightedPosition<short,3>::coordinates_type coords1;
    features::ObjectWeightedPosition<short,3>::values_type intens1;

    features::ObjectWeightedPosition<short,3>::coordinates_type coords2;
    features::ObjectWeightedPosition<short,3>::values_type intens2;

    features::ObjectWeightedPosition<short,3>::coordinates_type coordsM;
    features::ObjectWeightedPosition<short,3>::values_type intensM;

    for(int i = 0; i < 5; i++){
        for(int j = 0; j < 5; j++){
            for(int k = 0; k < 5; k++){
                coords1.push_back(vigra::MultiArrayShape<3>::type(i,j,k));
                intens1.push_back(short(i));
                coords2.push_back(vigra::MultiArrayShape<3>::type(i+5,j,k));
                intens2.push_back(short(1000-i));

                coordsM.push_back(vigra::MultiArrayShape<3>::type(i,j,k));
                intensM.push_back(short(i));
                coordsM.push_back(vigra::MultiArrayShape<3>::type(i+5,j,k));
                intensM.push_back(short(1000-i));
            }
        }
    }

    features::ObjectWeightedPosition<short,3> p1 (coords1, intens1);
    features::ObjectWeightedPosition<short,3> p2 (coords2, intens2);
    features::ObjectWeightedPosition<short,3> pM (coordsM, intensM);

    p1.mergeWith(p2);
    for(int i = 0; i < 12; i++){
        shouldEqualTolerance(p1.get()[i],pM.get()[i], 0.0000001);
    }
}



void testFeaturePrincipalComponents () {
    three_set coordinates;
    value_set values;

    feature_array pc = features::extractPrincipalComponents(coordinates,values);
    for(int i = 0; i < 12; i++)
        should(pc(i) == -1);

    coordinates.push_back( three_coordinate(1,1,1) );
    values.push_back(1);

    pc = features::extractPrincipalComponents(coordinates,values);
    for(int i = 0; i < 12; i++)
        should(pc(i) == -1);

    coordinates.push_back( three_coordinate(2,1,1) ); values.push_back(1);
    coordinates.push_back( three_coordinate(3,1,1) ); values.push_back(1);
    coordinates.push_back( three_coordinate(4,1,1) ); values.push_back(1);
    coordinates.push_back( three_coordinate(5,1,1) ); values.push_back(1);

    pc = features::extractPrincipalComponents(coordinates,values);
    shouldEqualTolerance(std::abs(pc(3)), 1, 0.001);
    shouldEqualTolerance(1-std::abs(pc(4)), 1, 0.001);
    shouldEqualTolerance(1-std::abs(pc(5)), 1, 0.001);


    coordinates.push_back( three_coordinate(1,2,1) ); values.push_back(1);
    coordinates.push_back( three_coordinate(2,2,1) ); values.push_back(1);
    coordinates.push_back( three_coordinate(3,2,1) ); values.push_back(1);
    coordinates.push_back( three_coordinate(4,2,1) ); values.push_back(1);
    coordinates.push_back( three_coordinate(5,2,1) ); values.push_back(1);

    pc = features::extractPrincipalComponents(coordinates,values);
    shouldEqualTolerance(std::abs(pc(3)), 1, 0.001);
    shouldEqualTolerance(1-std::abs(pc(4)), 1, 0.001);
    shouldEqualTolerance(1-std::abs(pc(5)), 1, 0.001);
    shouldEqualTolerance(1-std::abs(pc(6)), 1, 0.001);
    shouldEqualTolerance(std::abs(pc(7)), 1, 0.001);
    shouldEqualTolerance(1-std::abs(pc(8)), 1, 0.001);

    coordinates.push_back( three_coordinate(1,1,3) ); values.push_back(1);
    coordinates.push_back( three_coordinate(2,1,3) ); values.push_back(1);
    coordinates.push_back( three_coordinate(3,1,3) ); values.push_back(1);
    coordinates.push_back( three_coordinate(4,1,3) ); values.push_back(1);
    coordinates.push_back( three_coordinate(5,1,3) ); values.push_back(1);
    coordinates.push_back( three_coordinate(1,2,3) ); values.push_back(1);
    coordinates.push_back( three_coordinate(2,2,3) ); values.push_back(1);
    coordinates.push_back( three_coordinate(3,2,3) ); values.push_back(1);
    coordinates.push_back( three_coordinate(4,2,3) ); values.push_back(1);
    coordinates.push_back( three_coordinate(5,2,3) ); values.push_back(1);

    pc = features::extractPrincipalComponents(coordinates,values);
    shouldEqualTolerance(pc(3), 1, 0.001);
    shouldEqualTolerance(1-pc(4), 1, 0.001);
    shouldEqualTolerance(1-pc(5), 1, 0.001);
    shouldEqualTolerance(1-pc(6), 1, 0.001);
    shouldEqualTolerance(1-pc(7), 1, 0.001);
    shouldEqualTolerance(pc(8), 1, 0.001);
    shouldEqualTolerance(1-pc(9), 1, 0.001);
    shouldEqualTolerance(pc(10), 1, 0.001);
    shouldEqualTolerance(1-pc(11), 1, 0.001);


    // test if feature function and feature class produce same results
    features::ObjectPrincipalComponents<unsigned short,3> o ( coordinates,values);
    feature_array pc_obj = o.get();

    shouldEqual(pc_obj[0], pc[0]);
    shouldEqual(pc_obj[1], pc[1]);
    shouldEqual(pc_obj[2], pc[2]);
    shouldEqual(pc_obj[3], pc[3]);
    shouldEqual(pc_obj[4], pc[4]);
    shouldEqual(pc_obj[5], pc[5]);
    shouldEqual(pc_obj[6], pc[6]);
    shouldEqual(pc_obj[7], pc[7]);
    shouldEqual(pc_obj[8], pc[8]);
    shouldEqual(pc_obj[9], pc[9]);
    shouldEqual(pc_obj[10], pc[10]);
    shouldEqual(pc_obj[11], pc[11]);



}



void testFeatureIntensity () {
    three_set coordinates;
    value_set intensities;

    feature_array fint = features::extractIntensity(coordinates,intensities);
    for(int i = 0; i < 4; i++)
        should(fint(i) == -1);

    coordinates.push_back( three_coordinate(1,1,1) );
    intensities.push_back(1);

    fint = features::extractIntensity(coordinates,intensities);
    should(fint(0) == 1);
    for(int i = 1; i < 4; i++)
        should(fint(i) == -1);

    for(int i = 0; i < 4; i++){
        coordinates.push_back(three_coordinate(1,1,1));
        intensities.push_back(i+2);
    }

    fint = features::extractIntensity(coordinates,intensities);
    shouldEqualTolerance(fint(0), 3, 0.000001);
    shouldEqualTolerance(fint(1), 2, 0.000001);
    shouldEqualTolerance(fint(2), 0, 0.000001);
    shouldEqualTolerance(fint(3), 34./20.-3., 0.00001);


    // test if feature function and feature class produce same results
    features::ObjectIntensity<unsigned short,3> o ( coordinates,intensities);
    feature_array fint_obj = o.get();

    shouldEqual(fint_obj[0], fint[0]);
    shouldEqual(fint_obj[1], fint[1]);
    shouldEqual(fint_obj[2], fint[2]);
    shouldEqual(fint_obj[3]-3, fint[3]);

}



void testFeatureIntMinMax () {
    three_set coordinates;
    value_set intensities;

    feature_array fint = features::extractMinMaxIntensity(coordinates,intensities);
    for(int i = 0; i < 9; i++)
        should(fint(i) == -1);

    coordinates.push_back( three_coordinate(1,1,1) );
    intensities.push_back( 1);

    fint = features::extractMinMaxIntensity(coordinates,intensities);
    for(int i = 1; i < 9; i++)
        should(fint(i) == 1);

    for(int i = 0; i < 4; i++){
        coordinates.push_back(three_coordinate(1,1,1));
        intensities.push_back( i+2);
    }

    fint = features::extractMinMaxIntensity(coordinates,intensities);
    shouldEqualTolerance(fint(0), 1, 0.000001);
    shouldEqualTolerance(fint(1), 5, 0.000001);

    coordinates.clear();
    intensities.clear();
    for(int i = 0; i < 100; i++){
        coordinates.push_back(three_coordinate(1,1,1));
        intensities.push_back(i);
    }

    fint = features::extractMinMaxIntensity(coordinates,intensities);
    shouldEqualTolerance(fint(0), 0, 0.000001);
    shouldEqualTolerance(fint(1), 99, 0.000001);
    shouldEqualTolerance(fint(2), 5, 0.000001);
    shouldEqualTolerance(fint(3), 10, 0.000001);
    shouldEqualTolerance(fint(4), 20, 0.000001);
    shouldEqualTolerance(fint(5), 50, 0.000001);
    shouldEqualTolerance(fint(6), 80, 0.000001);
    shouldEqualTolerance(fint(7), 90, 0.000001);
    shouldEqualTolerance(fint(8), 95, 0.000001);


    // test if feature function and feature class produce same results
    features::ObjectMinMaxIntensity<unsigned short,3> o ( coordinates,intensities);
    feature_array fint_obj = o.get();

    shouldEqual(fint_obj[0], 0);
    shouldEqual(fint_obj[1], 99);
    shouldEqual(fint_obj[2], 5);
    shouldEqual(fint_obj[3], 10);
    shouldEqual(fint_obj[4], 25);
    shouldEqual(fint_obj[5], 50);
    shouldEqual(fint_obj[6], 75);
    shouldEqual(fint_obj[7], 90);
    shouldEqual(fint_obj[8], 95);

}



void testFeaturePairwise() {
    three_set coordinates;
    value_set intensities;

    feature_array pair = features::extractPairwise(coordinates,intensities);
    for(int i = 0; i < 4; i++)
        should(pair(i) == 0);

    coordinates.push_back( three_coordinate(1,1,1) );
    intensities.push_back(1);

    pair = features::extractPairwise(coordinates,intensities);
    for(int i = 1; i < 4; i++)
        should(pair(i) == 0);

    coordinates.push_back(three_coordinate(0,1,1));
    intensities.push_back( 3);

    pair = features::extractPairwise(coordinates,intensities);
    shouldEqualTolerance(pair(0), 1, 0.000001);
    shouldEqualTolerance(pair(1), 2, 0.000001);
    shouldEqualTolerance(1+pair(2), 1, 0.000001);
    shouldEqualTolerance(1+pair(3), 1, 0.000001);

    coordinates.push_back(three_coordinate(2,1,1));
    intensities.push_back( 2 );

    pair = features::extractPairwise(coordinates,intensities);
    shouldEqualTolerance(pair(0), 1, 0.000001);
    shouldEqualTolerance(pair(1), 5./3., 0.000001);
    shouldEqualTolerance(pair(2), 1./3., 0.000001);
    shouldEqualTolerance(pair(3), 1., 0.000001);


    // test if feature function and feature class produce same results
    features::ObjectPairwise<unsigned short,3> o ( coordinates,intensities);
    feature_array pair_obj = o.get();

    shouldEqual(pair_obj[0], pair[0]);
    shouldEqual(pair_obj[1], pair[1]);
    shouldEqual(pair_obj[2], pair[2]);
    shouldEqual(pair_obj[3], pair[3]);

}



void testFeatureSGF() {
    three_set coordinates;
    value_set intensities;

    feature_array sgf = features::extractSGF(coordinates,intensities);
    for(int i = 0; i < 48; i++)
        should(sgf(i) == 0);

    coordinates.push_back( three_coordinate(1,1,1) );
    intensities.push_back(5);

    sgf = features::extractSGF(coordinates,intensities);
    for(int i = 0; i < 48; i++)
        should(sgf(i) == 0);

    coordinates.clear();
    intensities.clear();
    coordinates.push_back( three_coordinate(0,0,0) ); intensities.push_back(1);
    coordinates.push_back( three_coordinate(0,0,1) ); intensities.push_back(1);
    coordinates.push_back( three_coordinate(0,1,0) ); intensities.push_back(5);
    coordinates.push_back( three_coordinate(0,1,1) ); intensities.push_back(5);
    coordinates.push_back( three_coordinate(0,2,0) ); intensities.push_back(1);
    coordinates.push_back( three_coordinate(0,2,1) ); intensities.push_back(1);

    sgf = features::extractSGF(coordinates,intensities);
    shouldEqualTolerance(sgf(0), 1./3., 0.000001);
    shouldEqualTolerance(sgf(1), 1./3., 0.000001);
    shouldEqualTolerance(sgf(2), 12.5, 0.000001);
    shouldEqualTolerance(sgf(3), 6.9221865524317288, 0.000001);

    shouldEqualTolerance(sgf(4), 0.27703200213814694, 0.000001);
    shouldEqualTolerance(sgf(5), 0.27703200213814694, 0.000001);
    shouldEqualTolerance(sgf(6), 12.5, 0.000001);
    shouldEqualTolerance(sgf(7), 6.9221865524317288, 0.000001);

    shouldEqualTolerance(sgf(8), 0.8090107968982081, 0.000001);
    shouldEqualTolerance(sgf(9), 0.8090107968982081, 0.000001);
    shouldEqualTolerance(sgf(10), 12.5, 0.000001);
    shouldEqualTolerance(sgf(11), 6.9221865524317288, 0.000001);

    shouldEqualTolerance(sgf(12), 1.6180215937964162, 0.000001);
    shouldEqualTolerance(sgf(13), 1.6180215937964162, 0.000001);
    shouldEqualTolerance(sgf(14), 12.5, 0.000001);
    shouldEqualTolerance(sgf(15), 6.9221865524317288, 0.000001);

    shouldEqualTolerance(sgf(16), 2./3., 0.000001);
    shouldEqualTolerance(sgf(17), 2./3., 0.000001);
    shouldEqualTolerance(sgf(18), 12.5, 0.000001);
    shouldEqualTolerance(sgf(19), 6.9221865524317288, 0.000001);

    shouldEqualTolerance(sgf(20), 2, 0.000001);
    shouldEqualTolerance(sgf(21), 2, 0.000001);
    shouldEqualTolerance(sgf(22), 12.5, 0.000001);
    shouldEqualTolerance(sgf(23), 6.9221865524317288, 0.000001);


    // test if feature function and feature class produce same results
    features::ObjectSGF<unsigned short,3> o ( coordinates,intensities);
    feature_array sgf_obj = o.get();

    for(int i = 0; i < 48; i++){
        shouldEqual(sgf_obj[i], sgf[i]);
    }

}



int main(){
    testFeatureVolume();
    testFeaturePosition();
    testFeatureBoundingBox();
    testFeatureWeightedPosition();
    testFeaturePrincipalComponents();
    testFeatureIntensity();
    testFeatureIntMinMax();
    testFeaturePairwise();
    testFeatureSGF();

    std::cout << "All tests passed." << std::endl;
    return 0;
}
