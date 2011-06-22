#ifndef __MX_MAT_FILE__
#define __MX_MAT_FILE__

#include <iostream>
#include <vigra/matrix.hxx>
//#include <mat.h>
//#include <vigra/matlab.hxx>

using namespace vigra;
//
//typedef struct tagOptions {
//	int nFeatureScales;
//	std::vector<std::vector<float > > sigmas;
//	std::vector<std::vector<float > > sigmas1;
//	std::vector<std::vector<float > > sigmas2;
//	std::vector<std::vector<float > > scales;
//	std::vector<float > interpolation;
//	float threshold;
//	std::string fileRF;
//	int nFeatures;
//} Options;
//

template<int DIM >
typename MultiArray<DIM, double >::difference_type string2shape(const std::string &str, const char sep=',') 
{
    typename MultiArray<DIM, double >::difference_type shape;
	std::vector<int > tokens;
	std::string tmp = str;
	size_t pos = tmp.find_first_of(sep);
    int iDim = 0;
	while (pos != std::string::npos) {
		shape[iDim ++] = atoi(tmp.substr(0, pos).c_str());
		tmp = tmp.substr(pos+1, tmp.length());
		pos = tmp.find_first_of(sep);
	}
    shape[iDim ++] = atoi(tmp.substr(0, pos).c_str());
    
	return shape;
}

std::vector<int > string2integers(const std::string &str, const char sep=',') 
{
	std::vector<int > tokens;
	std::string tmp = str;
	size_t pos = tmp.find_first_of(sep);
	while (pos != std::string::npos) {
		tokens.push_back(atoi(tmp.substr(0, pos).c_str()));
		tmp = tmp.substr(pos+1, tmp.length());
		pos = tmp.find_first_of(sep);
	}
	tokens.push_back(atoi(tmp.c_str()));

	return tokens;
}


std::vector<float > string2floats(const std::string &str, const char sep=',') 
{
	std::vector<float > tokens;
	std::string tmp = str;
	size_t pos = tmp.find_first_of(sep);
	while (pos != std::string::npos) {
		tokens.push_back(atof(tmp.substr(0, pos).c_str()));
		tmp = tmp.substr(pos+1, tmp.length());
		pos = tmp.find_first_of(sep);
	}
	tokens.push_back(atof(tmp.c_str()));

	return tokens;
}
//
//// function that converts a file to tokens
//std::vector<std::string > file2strings(const std::string &filename) {
//	std::ifstream ifs(filename.c_str());
//	std::vector<std::string > strings;
//	if (ifs) {
//		std::string file;
//        while(!ifs.eof()) {
//            ifs >> file;
//			if (file.length() > 0)
//				strings.push_back(file);
//        }
//		ifs.close();
//    }
//	std::cerr << "finish parsing file " << filename << std::endl;
//	return strings;
//}
//
//void parseFeatureParameters(Options &opt, std::string &str) {
//	std::vector<float > values = string2tokens(str, ',');
//	std::cerr << "parseFeatureParameters: token size=" << values.size() << std::endl;
//
//	std::vector<float > vec;
//	vec.resize(3);
//	// add sigmas
//	std::copy(values.begin(), values.begin()+3, vec.begin());
//	opt.sigmas.push_back(vec);
//	// add sigmas1
//	std::copy(values.begin()+3, values.begin()+6, vec.begin());
//	opt.sigmas1.push_back(vec);
//	// add sigmas2
//	std::copy(values.begin()+6, values.begin()+9, vec.begin());
//	opt.sigmas2.push_back(vec);
//	// add scales
//	std::copy(values.begin()+9, values.begin()+12, vec.begin());
//	opt.scales.push_back(vec);
//
//	opt.nFeatureScales ++;
//}
//
//template <class T >
//std::auto_ptr<RandomForest<T > > loadRandomForestMatFile(const std::string &fileRF) {
//	MATFile *pmat;
//	std::cerr << "opening file: " << fileRF << std::endl;
//	pmat = matOpen(fileRF.c_str(), "r");
//	mxArray *pa = matGetVariable(pmat, "random_forest");
//	if (!pa) {
//		std::cerr << "cannot find the variable: 'random_forest' " << std::endl;
//		return std::auto_ptr<RandomForest<T > >(0);
//	}
//	std::auto_ptr<RandomForest<T > > rf = std::auto_ptr<RandomForest<T > >(
//		matlab::importRandomForest<T >(matlab::getCellArray(pa)));
//
//	std::cerr << "finish loading random forest object" << std::endl;
//	std::cerr << "# of features " << rf->featureCount() << std::endl;
//	std::cerr << "# of trees " << rf->treeCount() << std::endl;
//
//	return rf;
//}
//
//Options file2options(const std::string &filename) {
//	std::ifstream ifs(filename.c_str());
//
//	// default options
//	Options opt;
//	opt.nFeatureScales = 0;
//	opt.fileRF = "";
//
//	if (ifs) {
//		std::string field, value;
//        while(!ifs.eof()) {
//            ifs >> field >> value;
//			if (field == "FeaturesParameters") 
//				parseFeatureParameters(opt, value);
//			else if (field == "RandomForestFile") 
//				opt.fileRF = value.c_str();
//			else if (field == "nFeatures") 
//				opt.nFeatures = atoi(value.c_str());
//			else if (field == "Interpolation") 
//				opt.interpolation = string2tokens(value);
//			else if (field == "Threshold") 
//				opt.threshold = atof(value.c_str());
//        }
//		ifs.close();
//    }
//
//	return opt;
//}
//
//void parseFileListFile(const std::string &filename, 
//					   std::vector<std::string > &srcFiles, 
//					   std::vector<std::string > &destFiles) 
//{
//	std::ifstream ifs(filename.c_str());
//
//	if (ifs) {
//        while(!ifs.eof()) {
//			std::string srcFile, destFile;
//            ifs >> srcFile >> destFile;
//			if (!srcFile.empty() && !destFile.empty()) {
//				srcFiles.push_back(srcFile);
//				std::cerr << "push back to srcFiles " << srcFile << std::endl;
//				destFiles.push_back(destFile);
//				std::cerr << "push back to destFiles " << destFile << std::endl;
//			}
//        }
//		ifs.close();
//    }
//}

/*
template <unsigned int DIM, class T >
MultiArrayView<DIM, T > mxArray2MultiArrayView(
	typename MultiArrayShape<DIM >::type const & shape, 
	mxArray * & t, 
	int dataSize = 4,
	bool createNew = false)
{
    mwSize matlabShape[DIM];
    for(int k=0; k<DIM; ++k)
        matlabShape[k] = static_cast<mwSize>(shape[k]);

	if (createNew) {
		if (dataSize == 4)
			t = mxCreateNumericArray(DIM, matlabShape, mxSINGLE_CLASS, mxREAL);
		else if (dataSize == 1)
			t = mxCreateNumericArray(DIM, matlabShape, mxLOGICAL_CLASS, mxREAL);		
	}

	return MultiArrayView<DIM, T>(shape, (T *)mxGetData(t));
}

template <unsigned int DIM, class T >
void mxReadFromMatFile(MultiArray<3, T> &vol,
	const std::string &matFile, 
	const std::string &fieldName) 
{
	MATFile *pmat;

	pmat = matOpen(matFile.c_str(), "r");
	if (pmat) {
		//std::cerr << "trying to get variable: " << fieldName << std::endl;
		mxArray *pa = matGetVariable(pmat, fieldName.c_str());
		//std::cerr << "getting dimensions: " << fieldName << std::endl;
		const mwSize *sz = mxGetDimensions(pa);
		typename MultiArrayShape< DIM >::type shape;
		if (DIM == 3) {
			//std::cerr << "mxArray object size: " << sz[0] << " " << sz[1] << " " << sz[2] << std::endl;
			shape = typename MultiArrayShape< DIM >::type(sz[0], sz[1], sz[2]);
		}
		else if (DIM == 4) {
			//std::cerr << "mxArray object size: " << sz[0] << " " << sz[1] << " " << sz[2] << " " << sz[3] << std::endl;
			shape = typename MultiArrayShape< DIM >::type(sz[0], sz[1], sz[2], sz[3]);
		}

		vol.reshape(shape);
		vol.copy(mxArray2MultiArrayView<DIM, T>(shape, pa));
		
		// clean up
		mxDestroyArray(pa);

		// close the file
		if (matClose(pmat) != 0) 
			std::cerr << "error closing file" << matFile << std::endl;
	}
	else {
		std::cerr << "error opening file " << matFile << std::endl;
	}
}

template <unsigned int DIM, class T >
bool mxWriteToMatFile(
	const MultiArrayView<DIM, T > &vol, 
	const std::string &matFile, 
	const std::string &fieldName) 
{
	MATFile *pmat;
	pmat = matOpen(matFile.c_str(), "wz");
	if (pmat) {
		mxArray *pa;
		MultiArrayView<DIM, T > dest = mxArray2MultiArrayView<DIM, T >(vol.shape(), pa, sizeof(vol[0]), true);
		dest.copy(vol);
		matPutVariable(pmat, fieldName.c_str(), pa);
		
		// close the file
		if (matClose(pmat) != 0) 
			std::cerr << "error closing file " << matFile << std::endl;
		
		// clean up
		mxDestroyArray(pa);
	}
	else {
		std::cerr << "error opening file: " << matFile << std::endl;
		return false;
	}

	return true;
}
*/
#endif /* __MX_MAT_FILE__ */
