#include "interfaces.hxx"



void processFile(std::string filename,
                 unsigned int features,
                 bool doLabeling,
                 std::string datasetname
                 )
{
    FileConfiguration fc (200,2);
    if(doLabeling){
        fc.LabelsPath = "labels";
        fc.SegmentationPath = datasetname;
    }else{
        fc.LabelsPath = datasetname;
        fc.SegmentationPath = datasetname;
    }

    TaskConfiguration tc (filename,false);
    tc.DoLabeling = doLabeling;
    tc.Features  = features;

    // label the connected components, if not already done
    segmentationLabelingAtOnce(fc,tc);

    // extract the features
    features::dFeatureExtractor f (filename,features,false);
    f.extract_features(features::FT_ALL);
}
