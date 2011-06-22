cmake_minimum_required(VERSION 2.6)
project( FEATURE_EXTRACTION )

OPTION(WITH_LEGACY_FEATURE_FORMAT "Backward support for the old feature format (many folders)" ON)

IF(WITH_LEGACY_FEATURE_FORMAT)
ADD_DEFINITIONS(-DWITH_LEGACY_FEATURE_FORMAT)
ENDIF(WITH_LEGACY_FEATURE_FORMAT)

OPTION(WITH_SEGMENTATION_LABELING "Execute segmentation labeling step before feature extraction (only needed once)" ON)

IF(WITH_SEGMENTATION_LABELING)
ADD_DEFINITIONS(-DWITH_SEGMENTATION_LABELING)
ENDIF(WITH_SEGMENTATION_LABELING)

OPTION(WITH_BOOSTMPI "Build feature extraction with MPI support. boost_mpi must be installed then." ON)


include_directories( include/ )
add_subdirectory(src)
add_subdirectory(examples)
