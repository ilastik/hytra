#
# (c) Bernhard X. Kausler, 2010
#

cmake_minimum_required(VERSION 2.6)
project( PYTHON_WRAPPER )

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake_extensions)

####################
### dependencies ###
####################

find_package( VIGRA REQUIRED )
find_package( Boost REQUIRED COMPONENTS python )
find_package( PythonLibs REQUIRED )

if(VIGRA_FOUND AND Boost_FOUND AND PYTHONLIBS_FOUND)
  include_directories(${VIGRA_INCLUDE_DIR} ${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
  link_directories( ${Boost_LIBRARY_DIRS} ${VIGRA_IMPEX_LIBRARY_DIR} ${PYTHON_LIBRARIES})
endif()

# assume by default that HDF5 is installed. Set definition for vigra random forest
ADD_DEFINITIONS(-DHasHDF5)



###############
### wrapper ###
###############

#!!!! segmentation wrapper is dysfunctional at the moment
#set(SEGMENTATION_SRCS segmentation/pySegmentation.cxx)
#include_directories( ../segmentation/include )

#add_library(pySegmentation SHARED ${SEGMENTATION_SRCS})
#target_link_libraries(pySegmentation ${Boost_PYTHON_LIBRARIES} hdf5 hdf5_hl boost_thread-mt ${VIGRA_NUMPY_CORE_LIBRARY})
#set_target_properties(pySegmentation PROPERTIES PREFIX "" OUTPUT_NAME "segmentation")
