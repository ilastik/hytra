#
# (c) Bernhard X. Kausler, 2010
#

#####################################################
### DELTA (Digital Embryos Lineage Tree Analyzer) ###
#####################################################

cmake_minimum_required(VERSION 2.6)
project(DELTA)

#######################
### code generation ###
#######################
ADD_CUSTOM_TARGET(hdf5FormatSpec ALL
     python ${DELTA_SOURCE_DIR}/hdf5_spec_generation/generateFileFormatSpecs.py  
       ${DELTA_SOURCE_DIR}/hdf5_spec_generation/hdf5_spec_v1.1.ini 
       ${DELTA_SOURCE_DIR}/empryonic/hdf5_format_spec.py  
       ${DELTA_SOURCE_DIR}/include/hdf5FormatSpec.h 
       "1.1"
     COMMENT "Generate hdf5 format specifications"
  )

##################
### components ###
##################
set(WITH_PYTHON_WRAPPER TRUE CACHE BOOL "Build Python bindings.")
set(WITH_SEGMENTATION TRUE CACHE BOOL "Build segmentation subproject.")
set(WITH_FEATURE_EXTRACTION TRUE CACHE BOOL "Build feature extraction subproject.")
set(WITH_TRACKING TRUE CACHE BOOL "Build cell tracking subproject.")

if(WITH_PYTHON_WRAPPER)
  add_subdirectory( python_wrapper )
endif()

if(WITH_SEGMENTATION)
  add_subdirectory( segmentation )
endif()

if(WITH_TRACKING)
  add_subdirectory( tracking )
endif()
