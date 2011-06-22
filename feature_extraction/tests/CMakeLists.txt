cmake_minimum_required(VERSION 2.6)

Project(TESTING-MYCLASS)

find_package( Boost 1.40.0 COMPONENTS unit_test_framework)

if(Boost_FOUND)
  add_definitions(-DBOOST_TEST_DYN_LINK)
  include_directories(${Boost_INCLUDE_DIRS} )

  ADD_EXECUTABLE(testSegmentationLabeling testSegmentationLabeling.cxx)
  TARGET_LINK_LIBRARIES(testSegmentationLabeling hdf5 hdf5_hl boost_mpi ${Boost_LIBRARIES} vigraimpex segmentationLabeling)

  ADD_EXECUTABLE(testSuperVoxels testSuperVoxels.cxx)
  TARGET_LINK_LIBRARIES(testSuperVoxels hdf5 hdf5_hl boost_mpi ${Boost_LIBRARIES} vigraimpex readSuperVoxels segmentationLabeling)

  ADD_EXECUTABLE(testFeatureExtractor testFeatureExtractor.cxx)
  TARGET_LINK_LIBRARIES(testFeatureExtractor hdf5 hdf5_hl boost_mpi ${Boost_LIBRARIES} vigraimpex readSuperVoxels segmentationLabeling featureExtractor)

endif()



