cmake_minimum_required(VERSION 2.6)

# create libraries
ADD_LIBRARY(features supervoxelFeatures.cxx)

ADD_LIBRARY(featureExtractor dFeatureExtractor.cxx)

ADD_LIBRARY(segmentationLabeling segmentationLabeling.cxx)

ADD_LIBRARY(mpiMS MpiMasterSlave.cxx)

ADD_LIBRARY(readSuperVoxels readSuperVoxels.cxx)

ADD_LIBRARY(argParser ArgParser.cpp)

ADD_LIBRARY(interfaces interfaces.cxx)

TARGET_LINK_LIBRARIES(featureExtractor readSuperVoxels features vigraimpex)
TARGET_LINK_LIBRARIES(interfaces readSuperVoxels features featureExtractor segmentationLabeling)

IF(WITH_BOOSTMPI)
  TARGET_LINK_LIBRARIES(mpiMS featureExtractor segmentationLabeling readSuperVoxels boost_mpi)
ENDIF(WITH_BOOSTMPI)
