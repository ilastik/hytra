/*
 * dFeatureExtractor.cxx
 *
 *  Created on: Mar 22, 2010
 *      Author: mlindner
 */

#include "dFeatureExtractor.hxx"

namespace features {

void getVolumeAttributes(vigra::MultiArray<1,const char*> & attr);
void getBboxAttributes(vigra::MultiArray<1,const char*> & attr);
void getPositionAttributes(vigra::MultiArray<1,const char*> & attr);
void getComAttributes(vigra::MultiArray<1,const char*> & attr);
void getPcAttributes(vigra::MultiArray<1,const char*> & attr);
void getIntensityAttributes(vigra::MultiArray<1,const char*> & attr);
void getIntMinMaxAttributes(vigra::MultiArray<1,const char*> & attr);
void getIntMaxPosAttributes(vigra::MultiArray<1,const char*> & attr);
void getPairAttributes(vigra::MultiArray<1,const char*> & attr);
void getSGFAttributes(vigra::MultiArray<1,const char*> & attr);

std::string dFeatureExtractor::extract_features(feature_flag features){
    if(verbose_){
        std::cout << "- Extracting features\n";
    }

    label_type max_label = r_.maxLabel(3);

    std::string output_file = filename_;

    vigra::HDF5File ofile(output_file,vigra::HDF5File::Open);

#ifdef WITH_LEGACY_FEATURE_FORMAT
    ofile.cd_mk("/features");

    // write datasets with general information

    // labelcount stores the total number of labels, both sane and zombie
    // labels. Note that the range of labels is 1 ... labelcount.
    ofile.write("labelcount",max_label);

    // labelcontent holds a 0 for each zombie label and a 1 for each sane label.
    // Zombie labels do not occur in the labeled segmentation and therefore do
    // not have any features. labelcontent[0] holds the information for label 1.
    label_array label_content = r_.labelContent();
    ofile.write("labelcontent",label_content);

    // featureconfig hold the configuration of features that were calculated.
    vigra::MultiArray<1,feature_flag> feature_config (array_shape(1));
    feature_config [0] = features;
    ofile.write("featureconfig",feature_config);
#endif

    // create feature matrices for the new standard (one matrix for each feature)
    feature_matrix volume     ( matrix_shape(1, max_label) );
    feature_matrix position   ( matrix_shape(12,max_label) );
    feature_matrix com        ( matrix_shape(12,max_label) );
    feature_matrix bbox       ( matrix_shape(7, max_label) );
    feature_matrix pc         ( matrix_shape(12,max_label) );
    feature_matrix intensity  ( matrix_shape(4, max_label) );
    feature_matrix intminmax  ( matrix_shape(9, max_label) );
    feature_matrix pair       ( matrix_shape(4, max_label) );
    feature_matrix sgf        ( matrix_shape(48,max_label) );
    feature_matrix intmaxpos  ( matrix_shape(4, max_label) );
    feature_matrix lcom       ( matrix_shape(12,max_label) );
    feature_matrix lpc        ( matrix_shape(12,max_label) );
    feature_matrix lintensity ( matrix_shape(4, max_label) );
    feature_matrix lintminmax ( matrix_shape(9, max_label) );
    feature_matrix lpair      ( matrix_shape(4, max_label) );
    feature_matrix lsgf       ( matrix_shape(48,max_label) );

    // extract&save features for every cell
    for(unsigned int i = 1; i <= max_label; i++){

        if(verbose_){
            std::cout << "  Processing label " << i << " of " << max_label << ".\n";
        }
        // load supervoxel data
        three_set set = getSuperVoxelCoordinates_(r_,i);
        feature_array val = getIntensityValues_(set);

#ifdef WITH_LEGACY_FEATURE_FORMAT
        ofile.cd("/features");
        char number[10];
        sprintf(number, "%i", i);
        ofile.cd_mk(number);
#endif

        // extract features if requested
        if(features & FT_VOLUME){
            feature_array ft = extractVolume(set, val);
#ifdef WITH_LEGACY_FEATURE_FORMAT
            ofile.write("volume",ft);
#endif
            volume.bind<1>(i-1) = ft;
        }
        if(features & FT_POSITION){
            feature_array ft = extractPosition(set, val);
#ifdef WITH_LEGACY_FEATURE_FORMAT
            ofile.write("position",ft);
#endif
            position.bind<1>(i-1) = ft;
        }
        if(features & FT_CENTER_OF_MASS){
            feature_array ft = extractWeightedPosition(set, val);
#ifdef WITH_LEGACY_FEATURE_FORMAT
            ofile.write("com",ft);
#endif
            com.bind<1>(i-1) = ft;
        }
        if(features & FT_BOUNDING_BOX){
            feature_array ft = extractBoundingBox(set, val);
#ifdef WITH_LEGACY_FEATURE_FORMAT
            ofile.write("bbox",ft);
#endif
            bbox.bind<1>(i-1) = ft;
        }
        if(features & FT_PRINCIPAL_COMPONENTS){
            feature_array ft = extractPrincipalComponents(set, val);
#ifdef WITH_LEGACY_FEATURE_FORMAT
            ofile.write("pc",ft);
#endif
            pc.bind<1>(i-1) = ft;
        }
        if(features & FT_INTENSITY){
            feature_array ft = extractIntensity(set, val);
#ifdef WITH_LEGACY_FEATURE_FORMAT
            ofile.write("intensity",ft);
#endif
            intensity.bind<1>(i-1) = ft;
        }
        if(features & FT_INTENSITY_MIN_MAX){
            feature_array ft = extractMinMaxIntensity(set, val);
#ifdef WITH_LEGACY_FEATURE_FORMAT
            ofile.write("intminmax",ft);
#endif
            intminmax.bind<1>(i-1) = ft;
        }
        // Experimental pairwise features.
        if(features & FT_EXPERIMENTAL_PAIR){
            feature_array ft = extractPairwise(set, val);
#ifdef WITH_LEGACY_FEATURE_FORMAT
            ofile.write("pair",ft);
#endif
            pair.bind<1>(i-1) = ft;
        }
        // experimental statistical geometric features SGF
        if(features & FT_EXPERIMENTAL_SGF){
            feature_array ft = extractSGF(set, val);
#ifdef WITH_LEGACY_FEATURE_FORMAT
            ofile.write("sgf",ft);
#endif
            sgf.bind<1>(i-1) = ft;
        }

        feature_array ft = extractMaxIntensity(set, val);
#ifdef WITH_LEGACY_FEATURE_FORMAT
        ofile.write("intmaxpos",ft);
#endif
        intmaxpos.bind<1>(i-1) = ft;

        // calculate features on large objects

        // load supervoxel data
        three_set lset = getLargeCoordinates_(r_,i);
        feature_array lval = getIntensityValues_(lset);

        if(features & FT_CENTER_OF_MASS){
            feature_array ft = extractWeightedPosition(lset, lval);
#ifdef WITH_LEGACY_FEATURE_FORMAT
            ofile.write("lcom",ft);
#endif
            lcom.bind<1>(i-1) = ft;
        }
        if(features & FT_PRINCIPAL_COMPONENTS){
            feature_array ft = extractPrincipalComponents(lset, lval);
#ifdef WITH_LEGACY_FEATURE_FORMAT
            ofile.write("lpc",ft);
#endif
            lpc.bind<1>(i-1) = ft;
        }
        if(features & FT_INTENSITY){
            feature_array ft = extractIntensity(lset, lval);
#ifdef WITH_LEGACY_FEATURE_FORMAT
            ofile.write("lintensity",ft);
#endif
            lintensity.bind<1>(i-1) = ft;
        }
        if(features & FT_INTENSITY_MIN_MAX){
            feature_array ft = extractMinMaxIntensity(lset, lval);
#ifdef WITH_LEGACY_FEATURE_FORMAT
            ofile.write("lintminmax",ft);
#endif
            lintminmax.bind<1>(i-1) = ft;
        }
        // Experimental pairwise features.
        if(features & FT_EXPERIMENTAL_PAIR){
            feature_array ft = extractPairwise(lset, lval);
#ifdef WITH_LEGACY_FEATURE_FORMAT
            ofile.write("lpair",ft);
#endif
            lpair.bind<1>(i-1) = ft;
        }
        // experimental statistical geometric features SGF
        if(features & FT_EXPERIMENTAL_SGF){
            feature_array ft = extractSGF(lset, lval);
#ifdef WITH_LEGACY_FEATURE_FORMAT
            ofile.write("lsgf",ft);
#endif
            lsgf.bind<1>(i-1) = ft;
        }

    }

    ofile.cd_mk("/objects/features");
    ofile.write("volume",volume);
    ofile.write("position",position);
    ofile.write("com",com);
    ofile.write("bbox",bbox);
    ofile.write("pc",pc);
    ofile.write("intensity",intensity);
    ofile.write("intminmax",intminmax);
    ofile.write("pair",pair);
    ofile.write("sgf",sgf);
    ofile.write("intmaxpos",intmaxpos);
    ofile.write("lcom",lcom);
    ofile.write("lpc",lpc);
    ofile.write("lintensity",lintensity);
    ofile.write("lintminmax",lintminmax);
    ofile.write("lpair",lpair);
    ofile.write("lsgf",lsgf);


    // write the attributes
    vigra::MultiArray<1,const char*> attr;

    getVolumeAttributes(attr);
    ofile.writeAttribute("volume","entries",attr);
    ofile.writeAttribute("volume","description","Total volume of the object");

    getPositionAttributes(attr);
    ofile.writeAttribute("position","entries",attr);
    ofile.writeAttribute("position","description","Mean position of object mask and higher central moments of distribution");

    getComAttributes(attr);
    ofile.writeAttribute("com","entries",attr);
    ofile.writeAttribute("com","description","Intensity weighted mean position of object and higher central moments of distribution");
    ofile.writeAttribute("lcom","entries",attr);
    ofile.writeAttribute("lcom","description","Intensity weighted mean position of enlarged bounding box and higher central moments of distribution");

    getBboxAttributes(attr);
    ofile.writeAttribute("bbox","entries",attr);
    ofile.writeAttribute("bbox","description","Minimum and maximum coordinates of object.");

    getPcAttributes(attr);
    ofile.writeAttribute("pc","entries",attr);
    ofile.writeAttribute("pc","description","Principal axes of inertia of the object");
    ofile.writeAttribute("lpc","entries",attr);
    ofile.writeAttribute("lpc","description","Principal axes of inertia of the enlarged bounding box");

    getIntensityAttributes(attr);
    ofile.writeAttribute("intensity","entries",attr);
    ofile.writeAttribute("intensity","description","Mean intensity and variance, skew and kurtosis of intensity distribution of object");
    ofile.writeAttribute("lintensity","entries",attr);
    ofile.writeAttribute("lintensity","description","Mean intensity and variance, skew and kurtosis of intensity distribution of enlarged bounding box");

    getIntMinMaxAttributes(attr);
    ofile.writeAttribute("intminmax","entries",attr);
    ofile.writeAttribute("intminmax","description","Quantiles of the intensity distribution in the object");
    ofile.writeAttribute("lintminmax","entries",attr);
    ofile.writeAttribute("lintminmax","description","Quantiles of the intensity distribution in the enlarged bounding box");

    getPairAttributes(attr);
    ofile.writeAttribute("pair","entries",attr);
    ofile.writeAttribute("pair","description","Energy measures between neighboring voxels");
    ofile.writeAttribute("lpair","entries",attr);
    ofile.writeAttribute("lpair","description","Energy measures between neighboring voxels");

    getSGFAttributes(attr);
    ofile.writeAttribute("sgf","entries",attr);
    ofile.writeAttribute("sgf","description","Statistical Geometric Features on object as described by Walker and Jackway (2002)");
    ofile.writeAttribute("lsgf","entries",attr);
    ofile.writeAttribute("lsgf","description","Statistical Geometric Features on enlarged bounding box as described by Walker and Jackway (2002)");

    getIntMaxPosAttributes(attr);
    ofile.writeAttribute("intmaxpos","entries",attr);
    ofile.writeAttribute("intmaxpos","description","Position of maximum intensity of the object");

    ofile.cd_mk("/objects/meta");

    label_array id (array_shape(max_label),label_type(0));
    label_array valid (array_shape(max_label),label_type(1));
    for(int i = 0; i < max_label; i++){
        id[i] = i+1;
    }
    ofile.write("id",id);
    ofile.write("valid",valid);



    return output_file;
}



feature_flag dFeatureExtractor::enable_features(feature_flag features){
	features_ = features_ | features;
	return features_;
}



feature_flag dFeatureExtractor::disable_features(feature_flag features){
	features_ = features_ & (FT_ALL - features);
	return features_;
}



std::string dFeatureExtractor::rawdata_hdf5_path_(){
    std::string path = PATH_RAW + "/" + NAME_RAW;
    return path;
}



three_set dFeatureExtractor::getSuperVoxelCoordinates_(GeometryReader& r, label_type cell_number){
	//load all coordinates that belong to cell cellNumber
	three_set Set = r.threeSet(cell_number);

	//convert from Topological coordinates to voxel coordinates (divide by 2)
	for(unsigned int i = 0; i < Set.size(); i++){
		for(int j = 0; j < 3; j++){
			Set[i][j] /= 2;
		}
	}

	return Set;
}



three_set dFeatureExtractor::getLargeCoordinates_(GeometryReader& r, label_type cell_number){
    //load all coordinates that belong to cell cellNumber
    three_set Set = r.threeSet(cell_number);

    // Find bounding box
    int min [3];
    int max [3];

    if(Set.size()>0)
    {
        min[0] = Set[0][0];
        min[1] = Set[0][1];
        min[2] = Set[0][2];
        max[0] = Set[0][0];
        max[1] = Set[0][1];
        max[2] = Set[0][2];
    }

    for(unsigned int i = 0; i < Set.size(); i++){
        for(int j = 0; j < 3; j++){
            if(Set[i][j] > max[j])
                max[j] = Set[i][j];
            if(Set[i][j] < min[j])
                min[j] = Set[i][j];
        }
    }


    for(int j = 0; j < 3; j++){
        max[j] /= 2;
        min[j] /= 2;
    }

    // resize, if size along one dimension is lower than sz
    int sz = 18;
    vigra::ArrayVector<hsize_t> shape = r.getDatasetShape(rawdata_hdf5_path_());
    for(int j = 0; j < 3; j++){
        int diff = max[j] - min[j];
        if(diff < sz){
            max[j] += std::ceil((sz-diff)/2);
            min[j] -= std::ceil((sz-diff)/2);
            // check if selection exceeds the boundary
            if(min[j]<0)
                min[j]=0;
            if(max[j]>=shape[j])
                max[j]=shape[j]-1;
        }
    }

    // create coordinate set
    three_set ret;
    for(int i = min[0]; i <= max[0]; i++){
        for(int j = min[1]; j <= max[1]; j++){
            for(int k = min[2]; k <= max[2]; k++){
                ret.push_back(three_coordinate(i,j,k));
            }
        }
    }

    return ret;
}



feature_array dFeatureExtractor::getIntensityValues_(three_set& coordinate_set){
	unsigned int size = coordinate_set.size();
	feature_array val (array_shape(size),0.);
	if(size == 0){
		return val;
	}

	// find min/max coordinate
	volume_shape min = coordinate_set[0], max = coordinate_set[0];
	for(three_iterator it = coordinate_set.begin(); it != coordinate_set.end(); ++it){
		if((*it)[0] < min[0]){
			min[0] = (*it)[0];
		}
		if((*it)[0] > max[0]){
			max[0] = (*it)[0];
		}
		if((*it)[1] < min[1]){
			min[1] = (*it)[1];
		}
		if((*it)[1] > max[1]){
			max[1] = (*it)[1];
		}
		if((*it)[2] < min[2]){
			min[2] = (*it)[2];
		}
		if((*it)[2] > max[2]){
			max[2] = (*it)[2];
		}
	}

	volume_shape block_size (max[0]-min[0]+1, max[1]-min[1]+1, max[2]-min[2]+1);

    // + read block from file
	raw_volume data (block_size);
    r_.readBlock(rawdata_hdf5_path_(), volume_shape(min[0],min[1],min[2]), block_size, data);


	//read the corresponding intensity for each coordinate
	for(unsigned int i = 0; i < size; i++){
		val[i] = feature_type(data(coordinate_set[i][0]-min[0],coordinate_set[i][1]-min[1],coordinate_set[i][2]-min[2]));
	}

	return val;
}


void getVolumeAttributes(vigra::MultiArray<1,const char*> & attr){
    attr.reshape(vigra::MultiArrayShape<1>::type(1));
    attr(0) = "Object volume, number of voxels";
}

void getBboxAttributes(vigra::MultiArray<1,const char*> & attr){
    attr.reshape(vigra::MultiArrayShape<1>::type(7));
    attr(0) = "Minimum x-coordinate found in object";
    attr(1) = "Minimum y-coordinate found in object";
    attr(2) = "Minimum z-coordinate found in object";
    attr(3) = "Maximum x-coordinate found in object";
    attr(4) = "Maximum y-coordinate found in object";
    attr(5) = "Maximum z-coordinate found in object";
    attr(6) = "Ratio (object volume)/(bounding box volume)";
}

void getPositionAttributes(vigra::MultiArray<1,const char*> & attr){
    attr.reshape(vigra::MultiArrayShape<1>::type(12));
    attr(0) = "Mean x-coordinate of the object";
    attr(1) = "Mean y-coordinate of the object";
    attr(2) = "Mean z-coordinate of the object";
    attr(3) = "Variance of x-coordinate distribution";
    attr(4) = "Variance of y-coordinate distribution";
    attr(5) = "Variance of z-coordinate distribution";
    attr(6) = "Skew of x-coordinate distribution";
    attr(7) = "Skew of y-coordinate distribution";
    attr(8) = "Skew of z-coordinate distribution";
    attr(9) = "Kurtosis of x-coordinate distribution";
    attr(10) = "Kurtosis of y-coordinate distribution";
    attr(11) = "Kurtosis of z-coordinate distribution";
}

void getComAttributes(vigra::MultiArray<1,const char*> & attr){
    attr.reshape(vigra::MultiArrayShape<1>::type(12));
    attr(0) = "Intensity-weighted Mean x-coordinate of the object";
    attr(1) = "Intensity-weighted Mean y-coordinate of the object";
    attr(2) = "Intensity-weighted Mean z-coordinate of the object";
    attr(3) = "Intensity-weighted Variance of x-coordinate distribution";
    attr(4) = "Intensity-weighted Variance of y-coordinate distribution";
    attr(5) = "Intensity-weighted Variance of z-coordinate distribution";
    attr(6) = "Intensity-weighted Skew of x-coordinate distribution";
    attr(7) = "Intensity-weighted Skew of y-coordinate distribution";
    attr(8) = "Intensity-weighted Skew of z-coordinate distribution";
    attr(9) = "Intensity-weighted Kurtosis of x-coordinate distribution";
    attr(10) = "Intensity-weighted Kurtosis of y-coordinate distribution";
    attr(11) = "Intensity-weighted Kurtosis of z-coordinate distribution";
}

void getPcAttributes(vigra::MultiArray<1,const char*> & attr){
    attr.reshape(vigra::MultiArrayShape<1>::type(12));
    attr(0) = "First principal moment of inertia of the object";
    attr(1) = "Second principal moment of inertia of the object";
    attr(2) = "Third principal moment of inertia of the object";
    attr(3) = "x-component of first principal axis of the object";
    attr(4) = "y-component of first principal axis of the object";
    attr(5) = "z-component of first principal axis of the object";
    attr(6) = "x-component of second principal axis of the object";
    attr(7) = "y-component of second principal axis of the object";
    attr(8) = "z-component of second principal axis of the object";
    attr(9) = "x-component of third principal axis of the object";
    attr(10) = "y-component of third principal axis of the object";
    attr(11) = "z-component of third principal axis of the object";
}

void getIntensityAttributes(vigra::MultiArray<1,const char*> & attr){
    attr.reshape(vigra::MultiArrayShape<1>::type(4));
    attr(0) = "Mean intensity";
    attr(1) = "Variance of intensity distribution";
    attr(2) = "Skew of intensity distribution";
    attr(3) = "Kurtosis of intensity distribution";
}

void getIntMinMaxAttributes(vigra::MultiArray<1,const char*> & attr){
    attr.reshape(vigra::MultiArrayShape<1>::type(9));
    attr(0) = "Minimum intensity in object";
    attr(1) = "Maximum intensity in object";
    attr(2) = "5% quantile of intensity distribution";
    attr(3) = "10% quantile of intensity distribution";
    attr(4) = "20% quantile of intensity distribution";
    attr(5) = "50% quantile of intensity distribution (median)";
    attr(6) = "80% quantile of intensity distribution";
    attr(7) = "90% quantile of intensity distribution";
    attr(8) = "95% quantile of intensity distribution";
}

void getPairAttributes(vigra::MultiArray<1,const char*> & attr){
    attr.reshape(vigra::MultiArrayShape<1>::type(4));
    attr(0) = "Mean absolute intensity difference of neighboring voxels";
    attr(1) = "Mean squared intensity difference of neighboring voxels";
    attr(2) = "Mean absolute symmetric difference quotient";
    attr(3) = "Mean absolute second derivative of intensity";
}

void getIntMaxPosAttributes(vigra::MultiArray<1,const char*> & attr){
    attr.reshape(vigra::MultiArrayShape<1>::type(4));
    attr(0) = "Maximum intensity of object";
    attr(1) = "x-coordinate of maximum intensity position";
    attr(2) = "y-coordinate of maximum intensity position";
    attr(3) = "z-coordinate of maximum intensity position";
}

void getSGFAttributes(vigra::MultiArray<1,const char*> & attr){
    attr.reshape(vigra::MultiArrayShape<1>::type(48));
    attr(0) = "Number of connected regions for background: maximum";
    attr(1) = "Number of connected regions for background: mean";
    attr(2) = "Number of connected regions for background: sample mean";
    attr(3) = "Number of connected regions for background: sample standard deviation";
    attr(4) = "Irregularity for background: maximum";
    attr(5) = "Irregularity for background: mean";
    attr(6) = "Irregularity for background: sample mean";
    attr(7) = "Irregularity for background: sample standard deviation";
    attr(8) = "Average clump displacement for background: maximum";
    attr(9) = "Average clump displacement for background: mean";
    attr(10) = "Average clump displacement for background: sample mean";
    attr(11) = "Average clump displacement for background: sample standard deviation";
    attr(12) = "Average clump inertia for background: maximum";
    attr(13) = "Average clump inertia for background: mean";
    attr(14) = "Average clump inertia for background: sample mean";
    attr(15) = "Average clump inertia for background: sample standard deviation";
    attr(16) = "Total clump area for background: maximum";
    attr(17) = "Total clump area for background: mean";
    attr(18) = "Total clump area for background: sample mean";
    attr(19) = "Total clump area for background: sample standard deviation";
    attr(20) = "Average clump area for background: maximum";
    attr(21) = "Average clump area for background: mean";
    attr(22) = "Average clump area for background: sample mean";
    attr(23) = "Average clump area for background: sample standard deviation";
    attr(24) = "Number of connected regions for foreground: maximum";
    attr(25) = "Number of connected regions for foreground: mean";
    attr(26) = "Number of connected regions for foreground: sample mean";
    attr(27) = "Number of connected regions for foreground: sample standard deviation";
    attr(28) = "Irregularity for foreground: maximum";
    attr(29) = "Irregularity for foreground: mean";
    attr(30) = "Irregularity for foreground: sample mean";
    attr(31) = "Irregularity for foreground: sample standard deviation";
    attr(32) = "Average clump displacement for foreground: maximum";
    attr(33) = "Average clump displacement for foreground: mean";
    attr(34) = "Average clump displacement for foreground: sample mean";
    attr(35) = "Average clump displacement for foreground: sample standard deviation";
    attr(36) = "Average clump inertia for foreground: maximum";
    attr(37) = "Average clump inertia for foreground: mean";
    attr(38) = "Average clump inertia for foreground: sample mean";
    attr(39) = "Average clump inertia for foreground: sample standard deviation";
    attr(40) = "Total clump area for foreground: maximum";
    attr(41) = "Total clump area for foreground: mean";
    attr(42) = "Total clump area for foreground: sample mean";
    attr(43) = "Total clump area for foreground: sample standard deviation";
    attr(44) = "Average clump area for foreground: maximum";
    attr(45) = "Average clump area for foreground: mean";
    attr(46) = "Average clump area for foreground: sample mean";
    attr(47) = "Average clump area for foreground: sample standard deviation";
}

} /* namespace fextract */
