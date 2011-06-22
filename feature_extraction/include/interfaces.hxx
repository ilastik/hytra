
#include "ConfigFeatures.hxx"
#include "segmentationLabeling.hxx"
#include "dFeatureExtractor.hxx"
#include "string"


/** Label connected components (optional) and extract features.
  *
  * filename:     pass the filename of the segmented HDF5 file here
  * features:     specify the set of features to be extracted
  * doLabeling:   set to false if the segmentation is already labeled
  * datasetname:  specify the dataset to process:
  *               doLabeling == true: path of the binary segmentation
  *               doLabeling == false: path of the labeled segmentation
  *
  * How to call:
  *
  * - Extract all features and label segmentation
  *
  *   processFile("/my/hdf5/file.h5", features::FT_ALL, true, "gc");
  *
  * - Extract center of mass position and volume only
  *
  *   processFile("/my/hdf5/file.h5", features::FT_CENTER_OF_MASS | features::FT_VOLUME, false, "whatever");
  *
  */
void processFile(std::string filename,
                 unsigned int features,
                 bool doLabeling,
                 std::string datasetname);

