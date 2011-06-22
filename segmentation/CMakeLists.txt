cmake_minimum_required(VERSION 2.6)

PROJECT(segmentation)

##
# defines
## 
# we need to add the following hdf5 api definitions explictly, because there is a bug in the ubuntu hdf5 package (11.04),
# that selects the wrong api version (see: https://bugs.launchpad.net/ubuntu/+source/hdf5/+bug/734889)
add_definitions(-DH5Dopen_vers=2 -DH5Gopen_vers=2 -DH5Gcreate_vers=2 -DH5Dcreate_vers=2 -DH5Acreate_vers=2)

#OPTION(BUILD_SHARED_LIBS OFF)

# Let the user choose an extra include path
SET(ADDITIONAL_INSTALL_PATH "~/local" CACHE PATH "Additional install path for local installations of vigra or hdf5")

# Specify the include directories here. Just adjust your local include directory.
INCLUDE_DIRECTORIES(
        ./src 
        extern/maxflow-v3.0
        ${ADDITIONAL_INSTALL_PATH}/include 
        /usr/include
)

# Specify where to search for libraries
LINK_DIRECTORIES(
        ${ADDITIONAL_INSTALL_PATH}/lib
)

# executable: single volume processing
ADD_EXECUTABLE(segmentation_single 
    ./apps/segmentation_single.cxx
    extern/maxflow-v3.0/graph.cpp
    extern/maxflow-v3.0/maxflow.cpp)

# executable: block processing for large volumes
#ADD_EXECUTABLE(segmentation_block
#    ./apps/segmentation_block.cxx
#    ../3rd_party/maxflow-v3.0/graph.cpp
#    ../3rd_party/maxflow-v3.0/maxflow.cpp)

# executable: block processing for large volumes, using the block processor class
ADD_EXECUTABLE(segmentation_block_processor
    ./apps/segmentation_block_processor.cxx
    extern/maxflow-v3.0/graph.cpp
    extern/maxflow-v3.0/maxflow.cpp)
    
# link all libraries to this executable (hdf5 (2x), vigra, own libraries)
TARGET_LINK_LIBRARIES(segmentation_single hdf5 hdf5_hl vigraimpex)
#TARGET_LINK_LIBRARIES(segmentation_block hdf5 hdf5_hl vigraimpex boost_thread-mt)
TARGET_LINK_LIBRARIES(segmentation_block_processor hdf5 hdf5_hl vigraimpex boost_thread-mt)

## Compiler flags
if(CMAKE_COMPILER_IS_GNUCXX)
    list(APPEND CMAKE_CXX_FLAGS -O3)
endif()

