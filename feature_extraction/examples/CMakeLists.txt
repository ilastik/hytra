cmake_minimum_required(VERSION 2.6)


IF(WITH_BOOSTMPI)
  # create executable for extractor (local mpi version)
  ADD_EXECUTABLE(extractor extractor.cxx)
  TARGET_LINK_LIBRARIES(extractor hdf5 hdf5_hl boost_mpi vigraimpex featureExtractor segmentationLabeling readSuperVoxels mpiMS argParser)

  # create executable for extractor_helics (helics mpi version)
  ADD_EXECUTABLE(extractor_helics extractor_helics.cxx)
  TARGET_LINK_LIBRARIES(extractor_helics hdf5 hdf5_hl boost_mpi vigraimpex featureExtractor segmentationLabeling readSuperVoxels mpiMS argParser)
ENDIF(WITH_BOOSTMPI)

# create executable for extractor_one (local non-mpi version)
ADD_EXECUTABLE(extractor_one extractor_one.cxx)
TARGET_LINK_LIBRARIES(extractor_one hdf5 hdf5_hl boost_mpi vigraimpex featureExtractor segmentationLabeling readSuperVoxels mpiMS argParser)


