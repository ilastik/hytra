#ifndef __CT_INI_CONFIGURATION__
#define __CT_INI_CONFIGURATION__

#include <iostream>
#include <vigra/matrix.hxx> 
#include "SimpleIni.h"
#include "HDF5ReaderWriter.hxx"

using namespace vigra;

#define INI_SECTION_DATA                    "DATA"
#define INI_SECTION_PREPROCESSING           "PRE_PROCESSING"
#define INI_SECTION_MULTISCALE_ANALYSIS     "MULTISCALE_ANALYSIS"
#define INI_SECTION_RUNTIME                 "RUNTIME"
#define INI_SECTION_SEEDED_WATERSHED        "SEEDED_WATERSHED"
#define INI_SECTION_FEATURE_EXTRACTION      "FEATURE_EXTRACTION"
#define INI_SECTION_RANDOM_FOREST           "RANDOM_FOREST"
#define INI_SECTION_GRAPH_CUT               "GRAPH_CUT"
#define INI_SECTION_BOUNDARY_CUE            "BOUNDARY_CUE"
#define INI_SECTION_POST_PROCESSING         "POST_PROCESSING"
#define INI_SECTION_BLOCK_PROCESSING        "BLOCK_PROCESSING"

std::string getHDF5File(const int frame, const CSimpleIniA &ini)
{
    // format the path    
    std::string strInFileTemplate = ini.GetValue(INI_SECTION_DATA, "path", "%04d.h5");
    char buf[1024];
    sprintf(buf, strInFileTemplate.c_str(), frame);
    return std::string(buf);
}

template<class T > void getMatrix(
        const CSimpleIniA &ini, 
        const std::string &section, 
        const std::string &variable, 
        Matrix<T >& matrix, 
        const std::string sep = ",") 
{
    // load the string
    std::string str = ini.GetValue(section.c_str(), variable.c_str(), "");
    if (str.empty())
        return ;
    
    // parse
	std::vector<T > tokens;
	std::string tmp = str;
	size_t pos = tmp.find_first_of(sep);
	while (pos != std::string::npos) {
		tokens.push_back(atof(tmp.substr(0, pos).c_str()));
		tmp = tmp.substr(pos+1, tmp.length());
		pos = tmp.find_first_of(sep);
	}
	tokens.push_back(static_cast<T >(atof(tmp.c_str())));

	matrix.reshape(typename Matrix<T >::size_type(1, tokens.size()), 0);
    std::copy(tokens.begin(), tokens.end(), matrix.begin());
}

#endif /* __CT_INI_CONFIGURATION__ */
