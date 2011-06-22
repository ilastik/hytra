
#ifndef CONFIGFEATURES_HXX
#define CONFIGFEATURES_HXX

#include <vector>
#include <string>

#include "vigra/multi_array.hxx"
#include "vigra/matrix.hxx"

// Datatypes used for raw data

typedef unsigned short raw_type;
typedef vigra::MultiArray<3,raw_type> raw_volume;


// Datatypes used for segmentation

typedef unsigned short seg_type;
typedef vigra::MultiArray<3,seg_type> seg_volume;

// Datatypes used for labeled volume

typedef unsigned short label_type;
typedef vigra::MultiArray<3,label_type> label_volume;


// Shape of 3D MultiArray

typedef vigra::MultiArrayShape<3>::type volume_shape;


// Datatype for coordinates

typedef short coordinate_type;

// 2D shape
typedef vigra::MultiArrayShape<2>::type matrix_shape;

// 1D MultiArrays / Lists

typedef vigra::MultiArray<1,label_type> label_array;  // arrays containing labels

typedef vigra::MultiArray<1,coordinate_type> dim_size_type; // arrays containing coordinate values

typedef float feature_type; // all feature values have this type
typedef vigra::MultiArray<1,feature_type> feature_array; // list of features

typedef vigra::MultiArrayShape<1>::type array_shape;

typedef vigra::MultiArray<2, feature_type> feature_matrix;


// Datatype for coordinate sets (vigra-based -> easy to read/write)

typedef vigra::MultiArray<2,coordinate_type> set_type;
typedef vigra::MultiArrayShape<2>::type set_shape;


// CGP-like datatypes for coordinate sets

typedef vigra::MultiArrayShape<3>::type three_coordinate;	// coordinate in 3d MultiArray
typedef std::vector<three_coordinate> three_set;	// List of 3D coordinates
typedef std::vector<three_coordinate>::iterator three_iterator;


// Datatype of parts counter in geometry file

typedef unsigned long ulong;
typedef vigra::MultiArray<1,ulong> counter_type;


// Configuration structures

/** HDF5 file configuration
  *
  * This structure contains all important information needed to process a HDF5 file.
  *
  * - Path to raw data (inside "/raw")
  * - Path to segmentation data (inside "/segmentation")
  * - Path to labeled segmentation (may be equal to segmentation data)
  * - Chunk size
  * - Compression parameter
  */
struct FileConfiguration
{
    FileConfiguration():
            RawPath("volume"),
            SegmentationPath("volume"),
            LabelsPath("labels"),
            CompressionParameter(2),
            ChunkSize(200,200,200) {};
    FileConfiguration(int chunksize, int compression):
            RawPath("volume"),
            SegmentationPath("volume"),
            LabelsPath("labels"),
            CompressionParameter(compression),
            ChunkSize(chunksize,chunksize,chunksize) {};
    FileConfiguration(std::string segpath, std::string lblpath,
                      int chunksize, int compression):
            RawPath("volume"),
            SegmentationPath(segpath),
            LabelsPath(lblpath),
            CompressionParameter(compression),
            ChunkSize(chunksize,chunksize,chunksize) {};


    std::string RawPath;
    std::string SegmentationPath;
    std::string LabelsPath;
    int CompressionParameter;
    volume_shape ChunkSize;
};



/** Task configuration
  *
  * - Path to HDF5 file to process
  * - Shape of the data blocks
  * - Features to extract
  * - Number of threads to be used
  * - Select tasks to do
  */
struct TaskConfiguration
{
    TaskConfiguration():
            Filename(""),
            BlockSize(200,200,200),
            Features(0xffffffff),
            Threads(1),
            DoLabeling(true),
            DoCollection(true),
            DoExtraction(true),
            Verbose(false) {};
    TaskConfiguration(std::string filename, bool verbose = false):
            Filename(filename),
            Features(0xffffffff),
            BlockSize(200,200,200),
            Threads(1),
            DoLabeling(true),
            DoCollection(true),
            DoExtraction(true),
            Verbose(verbose) {};
    TaskConfiguration(std::string filename, unsigned int blocksize,
                      bool verbose = false):
            Filename(filename),
            Features(0xffffffff),
            BlockSize(blocksize,blocksize,blocksize),
            Threads(1),
            DoLabeling(true),
            DoCollection(true),
            DoExtraction(true),
            Verbose(verbose) {};

    std::string Filename;
    volume_shape BlockSize;
    unsigned int Features;
    int Threads;
    bool DoLabeling;
    bool DoCollection;
    bool DoExtraction;
    bool Verbose;
};





#endif //CONFIGFEATURES_HXX
