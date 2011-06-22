cmake_minimum_required(VERSION 2.6)


project(object_features)

INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/../include)

ADD_LIBRARY(objectfeatures objectFeatures.cxx)

#VIGRA_ADD_NUMPY_MODULE(objectfeatures SOURCES
#        pythonwrapper.cpp
#        objectFeatures.cxx
#        #regressionForest.hpp
#
#LIBRARIES
#    ${VIGRANUMPY_LIBRARIES} features
#  VIGRANUMPY)
  
ADD_EXECUTABLE(objecttest test.cxx)
TARGET_LINK_LIBRARIES(objecttest objectfeatures vigraimpex)
