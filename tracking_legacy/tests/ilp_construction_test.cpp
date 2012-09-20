#define BOOST_TEST_MODULE ilp_construction_test

#include <vector>
#include <iostream>

#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <boost/shared_ptr.hpp>
#include <vigra/multi_array.hxx>

#include "energy.h"
#include "ilp_construction.h"

using namespace Tracking;
using namespace std;
using namespace boost;

BOOST_AUTO_TEST_CASE( AdaptiveEnergiesFormulation_constructor )
{
    double the_energy = 23;
    shared_ptr<ConstantEnergy> e(new ConstantEnergy(the_energy));
    
    //-----------------
    AdaptiveEnergiesFormulation a(e,e,e,e,72);
    BOOST_CHECK_EQUAL(a.Distance_threshold(), 72);
    
    a.Distance_threshold(36);
    BOOST_CHECK_EQUAL(a.Distance_threshold(), 36);

    AdaptiveEnergiesFormulation b( 49 );
    BOOST_CHECK_EQUAL(b.Distance_threshold(), 49);
}



BOOST_AUTO_TEST_CASE( AdaptiveEnergiesFormulation_formulate_ilp )
{
    // prepare mock objects
    Traxel t1, t2, t3;
    feature_array com1(feature_array::difference_type(3));
    feature_array com2(feature_array::difference_type(3));
    feature_array com3(feature_array::difference_type(3));

    com1[0] = 10;
    com1[1] = 11.3;
    com1[2] = 9.0;
    t1.Id = 27;
    t1.features["com"] = com1;

    com2[0] = 5;
    com2[1] = 4.5;
    com2[2] = 7;
    t2.Id = 12;
    t2.features["com"] = com2;

    com3[0] = 1;
    com3[1] = 0.5;
    com3[2] = 1.5;
    t3.Id = 3;
    t3.features["com"] = com3;

    Traxels prev, curr;
    prev[27] = t1;
    curr[12] = t2;
    curr[3] = t3;

    double the_energy = 23;
    shared_ptr<ConstantEnergy> e(new ConstantEnergy(the_energy));
    
    //-----------------
    AdaptiveEnergiesFormulation a;
    a.Division_energy(e).Move_energy(e).Disappearance_energy(e).Appearance_energy(e);

    pair<shared_ptr<AdaptiveEnergiesIlp>, vector<Event> > ret = a.formulate_ilp(prev, curr);
    
    // check the ilp
    //
    // | 1 1 1 |   | x_0->0   |    | 1 |
    // | 1 0 1 | x | x_0->1   | <= | 1 |
    // | 0 1 1 |   | x_0->0+1 |    | 1 |
    //
    IntegerLinearProgram ilp = *(ret.first);
    BOOST_CHECK_EQUAL(ilp.nVars, 3);
    BOOST_CHECK_EQUAL(ilp.nConstr, 3);
    BOOST_CHECK_EQUAL(ilp.nNonZero, 7);

    double rhs[] = {1,1,1};
    BOOST_CHECK_EQUAL_COLLECTIONS(ilp.rhs.begin(), ilp.rhs.end(), rhs, rhs+3);
    
    int matbeg[] = {0,2,4};
    BOOST_CHECK_EQUAL_COLLECTIONS(ilp.matbeg.begin(), ilp.matbeg.end(), matbeg, matbeg+sizeof(matbeg)/sizeof(int));

    int matcnt[] = {2,2,3};
    BOOST_CHECK_EQUAL_COLLECTIONS(ilp.matcnt.begin(), ilp.matcnt.end(), matcnt, matcnt+sizeof(matcnt)/sizeof(int));

    int matind[] = {0,1,0,2,0,1,2};
    BOOST_CHECK_EQUAL_COLLECTIONS(ilp.matind.begin(), ilp.matind.end(), matind, matind+sizeof(matind)/sizeof(int));

    double matval[] = {1,1,1,1,1,1,1};
    BOOST_CHECK_EQUAL_COLLECTIONS(ilp.matval.begin(), ilp.matval.end(), matval, matval+sizeof(matval)/sizeof(double));

    // check the events
    vector<Event> events = ret.second;
    Event move1, move2, division1;
    move1.type = Event::Move;
    move1.traxel_ids.push_back(27);
    move1.traxel_ids.push_back(3);
    move1.energy = the_energy;
    
    move2.type = Event::Move;
    move2.traxel_ids.push_back(27);
    move2.traxel_ids.push_back(12);
    move2.energy = the_energy;

    division1.type = Event::Division;
    division1.traxel_ids.push_back(27);
    division1.traxel_ids.push_back(3);
    division1.traxel_ids.push_back(12);
    division1.energy = the_energy;

    BOOST_CHECK_EQUAL(events.size(), 3);
    BOOST_CHECK(events[0] == move1 );
    BOOST_CHECK_EQUAL(events[0].energy, (the_energy - 2 * the_energy));
    BOOST_CHECK(events[1] == move2 );
    BOOST_CHECK_EQUAL(events[1].energy, (the_energy - 2 * the_energy));
    BOOST_CHECK(events[2] == division1 );
    BOOST_CHECK_EQUAL(events[2].energy, (the_energy - 3 * the_energy));
}



BOOST_AUTO_TEST_CASE( legacy_nearest_neighbor_search )
{
    // initialize some dummy traxels
    Traxel traxel1, traxel2, query;
    feature_array com1(feature_array::difference_type(3));
    feature_array com2(feature_array::difference_type(3));
    feature_array com_query(feature_array::difference_type(3));

    com1[0] = 10;
    com1[1] = 11.3;
    com1[2] = 9.0;
    traxel1.Id = 27;
    traxel1.features["com"] = com1;

    com2[0] = 5;
    com2[1] = 4.5;
    com2[2] = 7;
    traxel2.Id = 12;
    traxel2.features["com"] = com2;

    com_query[0] = 1;
    com_query[1] = 0.5;
    com_query[2] = 1.5;
    query.Id = 3;
    query.features["com"] = com_query;

    Traxels traxels;
    traxels[27] = traxel1;
    traxels[12] = traxel2;

    // test the nn search
    legacy::NearestNeighborSearch nn( traxels, "com", 3);

    // knn_in_range()
    map<unsigned int, double> ret;
    ret = nn.knn_in_range( query, 100, 2);
    BOOST_CHECK_EQUAL( ret.size(), 2 );
    BOOST_CHECK_CLOSE( ret[27], 15.93*15.93, 0.1 );
    BOOST_CHECK_CLOSE( ret[12], 7.89*7.89, 0.1 );

    ret = nn.knn_in_range( query, 0, 10);
    BOOST_CHECK_EQUAL( ret.size(), 0 );

    ret = nn.knn_in_range( query, 100, 0);
    BOOST_CHECK_EQUAL( ret.size(), 0 );

    ret = nn.knn_in_range( query, 100, 1);
    BOOST_CHECK_EQUAL( ret.size(), 1 );
    BOOST_CHECK_CLOSE( ret[12], 7.89*7.89, 0.1 );

    ret = nn.knn_in_range( query, 8, 2);
    BOOST_CHECK_EQUAL( ret.size(), 1 );
    BOOST_CHECK_CLOSE( ret[12], 7.89*7.89, 0.1 );

    // count_in_range()
    int c;
    c = nn.count_in_range( query, 100 );
    BOOST_CHECK_EQUAL( c, 2 );

    c = nn.count_in_range( query, 8 );
    BOOST_CHECK_EQUAL( c, 1 );

    c = nn.count_in_range( query, 1 );
    BOOST_CHECK_EQUAL( c, 0 );
}



BOOST_AUTO_TEST_CASE( feature2 )
{
    BOOST_CHECK_EQUAL( 23, 23 );
    BOOST_CHECK( true );
    BOOST_CHECK_CLOSE( 1.0 - 0.6, 0.4, 0.1);
}

// EOF

