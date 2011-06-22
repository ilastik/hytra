/*
 *  pythonwrapper.cpp
 *  vigra
 *
 *  Created by Luca Fiaschi on 3/10/11.
 *  Copyright 2011 Heidelberg university. All rights reserved.
 *
 */


#include "objectFeatures.hxx"
#include "ConfigFeatures.hxx"
#include <Python.h>
#include <boost/python.hpp>
#include <vigra/numpy_array.hxx>
#include <vigra/numpy_array_converters.hxx>
#include "helpers.hxx"

using namespace vigra;
namespace python = boost::python;


/*helper namespace*/

namespace details{
	
	
	template< class T >
	void NumpyArr2CoordArray (const vigra::NumpyArray< 2,T >& init_c , three_set& res_c){
		
		
		vigra_precondition(init_c.shape(1) ==3, "The input array should be Nx3 -> 3dim points.");
		
		res_c.reserve(init_c.shape(0));
		
		MultiArrayShape<2>::type p(0,0);
		
		three_coordinate temp;
		
		
		
		
		
		
		
		for (p[0]=0; p[0]<init_c.shape(0); ++p[0]) {
			temp[0]=init_c(int(p[0]),0);
			temp[1]=init_c(int(p[0]),1);
			temp[2]=init_c(int(p[0]),2);
			
			res_c.push_back(temp); 
		}
	
	}
	

    template< class T2 >
    void NumpyArr2IntensityArray (const vigra::NumpyArray <1,T2>& init_c , value_set& res_c){


        vigra_precondition(init_c.shape(1) ==1, "The input array should be Nx1.");

        res_c.reserve(init_c.shape(0));

        MultiArrayShape<2>::type p(0,0);


        for (p[0]=0; p[0]<init_c.shape(0); ++p[0]) {
            res_c.push_back(init_c(int(p[0]),0));
        }

    }
	
	template < class T2 >
	void FeatureArray2NumpyArray(const feature_array& F , vigra::NumpyArray <1,T2>& res=python::object()){
		
		
		
		MultiArrayShape<1>::type temp(F.size());
		
		res.reshapeIfEmpty(temp);
		
		MultiArrayShape<1>::type p(MultiArrayShape<1>::type::value_type(0));
		
		
		for(p[0]=0; p[0]<F.size(); ++p[0]){
					
			res[p]=F[p[0]];
			
		}
	
	}
	
	
	
	template < class T2>
	void NumpyArray2FeatureArray(const vigra::NumpyArray <1,T2>& intensities , feature_array& res){
		
		
		
		res.reshape(intensities.shape());
		
		MultiArrayShape<1>::type p(MultiArrayShape<1>::type::value_type(0));
		
		
		
		for(p[0]=0; p[0]<intensities.shape(0); ++p[0]){
					
			res[p]=intensities[p];
			
			
		}
		
	}
	
		
	
	
}



/*Here starts the wrapping of the functions*/


template < class T, class T2 >
NumpyAnyArray pythonExtractVolume(NumpyArray<2, T > coord, NumpyArray <1 ,T2> res=python::object() ){
	
	{ PyAllowThreads _pythread;
	
	
	three_set coordinates;
	
    value_set intensities;
	feature_array features;
	
	
	details::NumpyArr2CoordArray(coord,coordinates); //copy the numpy arrary into coordinates
	
	features=features::extractVolume(coordinates,intensities);
	
	details::FeatureArray2NumpyArray(features,res);
	
	return res;
	}
}


template < class T, class T2 >
NumpyAnyArray pythonExtractPosition(NumpyArray<2, T > coord, NumpyArray <1,T2> res=python::object() ){
	
	{ PyAllowThreads _pythread;
	
	three_set coordinates;
	
    value_set intensities;
	
	feature_array features;
	
	
	details::NumpyArr2CoordArray(coord,coordinates); //copy the numpy arrary into coordinates
	
	features=features::extractPosition(coordinates,intensities);
	
	details::FeatureArray2NumpyArray(features,res);
	
	
	return res;
	}
}



template < class T, class T2 >
NumpyAnyArray pythonExtractWeightedPosition(NumpyArray<2, T > coord, NumpyArray<1,T2> inten, NumpyArray <1,T2> res=python::object() ){
	
	{ PyAllowThreads _pythread;
		
	vigra_precondition( coord.shape(0)==inten.shape(0), "The input array should be Nx1 -> intensities" ); 
	
	
	three_set coordinates;
	
	details::NumpyArr2CoordArray(coord,coordinates); //copy the numpy arrary into coordinates
	
    value_set intensities;
    details::NumpyArr2IntensityArray(inten,intensities); //copy the numpy arrary into intensities

	
	
	feature_array features;
	
	
	features=features::extractWeightedPosition(coordinates,intensities);
	
	details::FeatureArray2NumpyArray(features,res);
	
	
	return res;
	}
}






template < class T, class T2 >
NumpyAnyArray pythonExtractPrincipalComponents(NumpyArray<2, T > coord, NumpyArray<1,T2> inten, NumpyArray <1,T2> res=python::object() ){
	
	{ PyAllowThreads _pythread;
	
	vigra_precondition( coord.shape(0)==inten.shape(0), "The input array should be Nx1 -> intensities" ); 
	
	
	three_set coordinates;
	
	details::NumpyArr2CoordArray(coord,coordinates); //copy the numpy arrary into coordinates
	
    value_set intensities;
    details::NumpyArr2IntensityArray(inten,intensities); //copy the numpy arrary into intensities
	
	
	feature_array features;
	
	
	features=features::extractPrincipalComponents(coordinates,intensities);  //THE APPLIED FUNCTION
	
	details::FeatureArray2NumpyArray(features,res);   //copy the result into numpy array
	
	
	return res;
	}
}



template < class T, class T2 >
NumpyAnyArray pythonExtractBoundingBox(NumpyArray<2, T > coord, NumpyArray<1,T2> inten, NumpyArray <1,T2> res=python::object() ){
	
	{ PyAllowThreads _pythread;
	
	vigra_precondition( coord.shape(0)==inten.shape(0), "The input array should be Nx1 -> intensities" ); 
	
	
	three_set coordinates;
	
	details::NumpyArr2CoordArray(coord,coordinates); //copy the numpy arrary into coordinates
	
    value_set intensities;
    details::NumpyArr2IntensityArray(inten,intensities); //copy the numpy arrary into intensities
	
	
	feature_array features;
	
	
	features=features::extractBoundingBox(coordinates,intensities);
	
	details::FeatureArray2NumpyArray(features,res);
	
	
	return res;
	}
}




template < class T, class T2 >
NumpyAnyArray pythonExtractStatisticsIntensity(NumpyArray<2, T > coord, NumpyArray<1,T2> inten, NumpyArray <1,T2> res=python::object() ){
	
	{ PyAllowThreads _pythread;
	
	vigra_precondition( coord.shape(0)==inten.shape(0), "The input array should be Nx1 -> intensities" ); 
	
	
	three_set coordinates;
	
	details::NumpyArr2CoordArray(coord,coordinates); //copy the numpy arrary into coordinates
	
    value_set intensities;
    details::NumpyArr2IntensityArray(inten,intensities); //copy the numpy arrary into intensities
	
	
	feature_array features;
	
	
	features=features::extractIntensity(coordinates,intensities);
	
	details::FeatureArray2NumpyArray(features,res);  //copy the result back into numpy arrays
	
	
	return res;
	}
}


template < class T, class T2 >
NumpyAnyArray pythonExtractMinMaxIntensity(NumpyArray<2, T > coord, NumpyArray<1,T2> inten, NumpyArray <1,T2> res=python::object() ){
	
	{ PyAllowThreads _pythread;
	
	vigra_precondition( coord.shape(0)==inten.shape(0), "The input array should be Nx1 -> intensities" ); 
	
	
	three_set coordinates;
	
	details::NumpyArr2CoordArray(coord,coordinates); //copy the numpy arrary into coordinates array
	
    value_set intensities;
    details::NumpyArr2IntensityArray(inten,intensities); //copy the numpy arrary into intensities
	
	
	feature_array features;
	
	
	features=features::extractMinMaxIntensity(coordinates,intensities);
	
	details::FeatureArray2NumpyArray(features,res);  //copy the result back into numpy arrays
	
	
	return res;
	}
}



template < class T, class T2 >
NumpyAnyArray pythonExtractMaxIntensity(NumpyArray<2, T > coord, NumpyArray<1,T2> inten, NumpyArray <1,T2> res=python::object() ){
	
	{ PyAllowThreads _pythread;
		
	vigra_precondition( coord.shape(0)==inten.shape(0), "The input array should be Nx1 -> intensities" ); 
	
	
	three_set coordinates;
	
	details::NumpyArr2CoordArray(coord,coordinates); //copy the numpy arrary into coordinates array
	
    value_set intensities;
    details::NumpyArr2IntensityArray(inten,intensities); //copy the numpy arrary into intensities
	
	
	feature_array features;
	
	
	features=features::extractMaxIntensity(coordinates,intensities);
	
	details::FeatureArray2NumpyArray(features,res);  //copy the result back into numpy arrays
	
	
	return res;
	}
}


template < class T, class T2 >
NumpyAnyArray pythonExtractPairwise(NumpyArray<2, T > coord, NumpyArray<1,T2> inten, NumpyArray <1,T2> res=python::object() ){
	
	{ PyAllowThreads _pythread;
	vigra_precondition( coord.shape(0)==inten.shape(0), "The input array should be Nx1 -> intensities" ); 
	
	
	three_set coordinates;
	
	details::NumpyArr2CoordArray(coord,coordinates); //copy the numpy arrary into coordinates array
	
    value_set intensities;
    details::NumpyArr2IntensityArray(inten,intensities); //copy the numpy arrary into intensities
	
	
	feature_array features;
	
	
	features=features::extractPairwise(coordinates,intensities);
	
	details::FeatureArray2NumpyArray(features,res);  //copy the result back into numpy arrays
	
	
	return res;
	}
}




template < class T, class T2 >
NumpyAnyArray pythonExtractHOG(NumpyArray<2, T > coord, NumpyArray<1,T2> inten, NumpyArray <1,T2> res=python::object() ){
	
	{ PyAllowThreads _pythread;
	
	vigra_precondition( coord.shape(0)==inten.shape(0), "The input array should be Nx1 -> intensities" ); 
	
	
	three_set coordinates;
	
	details::NumpyArr2CoordArray(coord,coordinates); //copy the numpy arrary into coordinates array
	
    value_set intensities;
    details::NumpyArr2IntensityArray(inten,intensities); //copy the numpy arrary into intensities
	
	
	feature_array features;
	
	
	features=features::extractHOG(coordinates,intensities);
	
	details::FeatureArray2NumpyArray(features,res);  //copy the result back into numpy arrays
	
	
	return res;
	}
}



template < class T, class T2 >
NumpyAnyArray pythonExtractSGF(NumpyArray<2, T > coord, NumpyArray<1,T2> inten, NumpyArray <1,T2> res=python::object() ){
	
	{ PyAllowThreads _pythread;
	
	vigra_precondition( coord.shape(0)==inten.shape(0), "The input array should be Nx1 -> intensities" ); 
	
	
	three_set coordinates;
	
	details::NumpyArr2CoordArray(coord,coordinates); //copy the numpy arrary into coordinates array
	
    value_set intensities;
    details::NumpyArr2IntensityArray(inten,intensities); //copy the numpy arrary into intensities
	
	
	feature_array features;
	
	
	features=features::extractSGF(coordinates,intensities);
	
	details::FeatureArray2NumpyArray(features,res);  //copy the result back into numpy arrays
	
	
	return res;
	}
}




template <class T >
boost::python::dict
pythonExtractSelectedFeature(NumpyArray<3, T > labeled_image, NumpyArray<3, float> image, std::string feature_name, std::string background){
	
	// Exctract the functor for the function
	
    feature_array (*fp)(three_set&,value_set&)=NULL;
	
		if (feature_name=="Volume") {
			fp=&features::extractVolume;
		}else if (feature_name== "Statistics Positions") {
			fp=&features::extractPosition;
		}else if (feature_name== "Statistics Weighted Positions") {
			fp=&features::extractWeightedPosition;
		}else if (feature_name== "Principal Components") {
			fp=&features::extractPrincipalComponents;
		}else if (feature_name== "Bounding Box") {
			fp=&features::extractBoundingBox;
		}else if (feature_name== "Statistics Intensity") {
			fp=&features::extractIntensity;
		}else if (feature_name== "MinMax Intensity") {
			fp=&features::extractMinMaxIntensity;
		}else if (feature_name== "Pairwise"){
			fp=&features::extractPairwise;
		}else if (feature_name=="HOG") {
			fp=&features::extractHOG;
		}else if (feature_name=="SGF") {
			std::cout << "WARNING still  experimental" << std::endl;
			fp=&features::extractSGF;
		} else{
			throw std::domain_error("Unknown Feature, read the help for the available ones!");
		}
	
	
	
	{ PyAllowThreads _pythread;
		
		
		MultiArrayShape<3>::type shape(image.shape());
		
		
		vigra_precondition(shape==labeled_image.shape(), " the labeled image and the intensity image must have the same dimension");
		
		
		
		//Create the map of label positions
		
		typename std::map< T , three_set > dpos;
		
		//Create the map of the intensities
		
		typedef std::vector<float*> feature_vector;
		typename std::map< T , feature_vector > dint;
		
		
		
		MultiArrayShape<3>::type p(0,0,0);
		
		
		T label;
		
		for (p[2]=0; p[2]<shape[2]; ++p[2]){
			for(p[1]=0; p[1]<shape[1]; ++p[1])
			{
				for(p[0]=0; p[0]<shape[0]; ++p[0])
				{	label=labeled_image[p];
					if (background=="Background") {
						dpos[label].push_back(p);
						dint[label].push_back(&image[p]);
					} else if (label!=0.) {                     //Skip background
						dpos[label].push_back(p);
						dint[label].push_back(&image[p]);

					} 						

					
				}
				
				
			}
		}
		
		
		//Very stupid conversion to dictionry of <T, feature_array>
		typename std::map< T , feature_vector>::iterator it_dint=dint.begin();
		typename std::map< T , feature_vector>::iterator end_dint=dint.end();
        typename std::map< T , value_set > dint_array;
		
		
        value_set* pointer;
		
		
		while (it_dint!=end_dint) {
			
			//std::cout<< "HERE" << std::endl;
            pointer=new value_set;
			//std::cout << (it_dint->second).size()<< std::endl;
			//std::cout << (it_dint->first)<< std::endl;
			
            //(*pointer).reshape(array_shape((it_dint->second).size()));
			//std::cout<< "before the loop" << std::endl;						  
			for(int i=0; i<(it_dint->second).size();i++){
				
				//std::cout<< "HERE inside the loop" << std::endl;
				//std::cout<< (*pointer).shape(0)<< " " << (*pointer).shape(1) << std::endl;
				
				//std::cout<< "valore intensity" << std::endl;
				//std::cout<< *((it_dint->second)[i])<< std::endl;

				
                //(*pointer)[i]=*((it_dint->second)[i]);
                (*pointer).push_back(*((it_dint->second)[i]));
				//std::cout<<(*pointer)[i,0]<< std::endl;
			}
			
			dint_array[it_dint->first]=*pointer;
				
			++it_dint;
		}

		//std::cout<< "Passed the point" << std::endl;
		//compute the functor for each supevoxel
		typename std::map< T , three_set >::iterator it_pos=dpos.begin();
		typename std::map< T , three_set >::iterator end_pos=dpos.end();
        typename std::map< T , value_set >::iterator it_dint_array=dint_array.begin();
		
		
		boost::python::dict pyd; //result dictionary
		
		NumpyArray<1,float>* res;
	
		while (it_pos!=end_pos) {
						
			res = new NumpyArray<1,float>;
			
			
			
			details::FeatureArray2NumpyArray((*fp)(it_pos->second,it_dint_array->second),*res);
			pyd[it_pos->first]=*res;
			
			
			//std::cout << "HERE"<< std::endl;
				
			++it_dint_array;
			++it_pos;
		
		}
		
			
		return pyd;
	}
}

template <class T >
NumpyAnyArray
pythonExtractSelectedFeature2Matrix(NumpyArray<3, T > labeled_image, NumpyArray<3, float> image, std::string feature_name){
	
	//itialize result matrix
	
	NumpyArray<2,float> res;
	
	//initialize the result shape
	MultiArrayShape<2>:: type res_shape(0,0);
	
	// Exctract the functor for the function
    feature_array (*fp)(three_set&,value_set&)=NULL;
	
	if (feature_name=="Volume") {
		fp=&features::extractVolume;
		res_shape[1]=1;
	}else if (feature_name== "Statistics Positions") {
		fp=&features::extractPosition;
		res_shape[1]=12;
	}else if (feature_name== "Statistics Weighted Positions") {
		fp=&features::extractWeightedPosition;
		res_shape[1]=12;
	}else if (feature_name== "Principal Components") {
		fp=&features::extractPrincipalComponents;
		res_shape[1]=12;
	}else if (feature_name== "Bounding Box") {
		fp=&features::extractBoundingBox;
		res_shape[1]=7;
	}else if (feature_name== "Statistics Intensity") {
		fp=&features::extractIntensity;
		res_shape[1]=4;
	}else if (feature_name== "Max Intensity") {
		fp=&features::extractMaxIntensity;
		res_shape[1]=4;
	}else if (feature_name== "MinMax Intensity") {
		fp=&features::extractMinMaxIntensity;
		res_shape[1]=9;
	}else if (feature_name== "Pairwise"){
		fp=&features::extractPairwise;
		res_shape[1]=4;
	}else if (feature_name=="HOG") {
		fp=&features::extractHOG;
		res_shape[1]=20;
	}else if (feature_name=="SGF") {
		std::cout << "WARNING still  experimental" << std::endl;
		fp=&features::extractSGF;
		res_shape[1]=48;
	} else{
		throw std::domain_error("Unknown Feature, read the help for the available ones!");
	}
	
	
	
	{ PyAllowThreads _pythread;
		
		
		MultiArrayShape<3>::type shape(image.shape());
		
		
		vigra_precondition(shape==labeled_image.shape(), " the labeled image and the intensity image must have the same dimension");
		
		
		
		//Create the map of label positions
		
		typename std::map< T , three_set > dpos;
		
		//Create the map of the intensities
		
		typedef std::vector<float*> feature_vector;
		typename std::map< T , feature_vector > dint;
		
		
		
		MultiArrayShape<3>::type p(0,0,0);
		
		
		T label;
		
		
		
		for (p[2]=0; p[2]<shape[2]; ++p[2]){
			for(p[1]=0; p[1]<shape[1]; ++p[1])
			{
				for(p[0]=0; p[0]<shape[0]; ++p[0])
				{	label=labeled_image[p];
			       	dpos[label].push_back(p);
					dint[label].push_back(&image[p]);
						
					} 						
										
				}
							
			}
		
		
		
		
	
	
		//Very stupid conversion to dictionry of <T, feature_array>
		//check that there are not missing labels
		typename std::map< T , feature_vector>::iterator it_dint=dint.begin();
		typename std::map< T , feature_vector>::iterator end_dint=dint.end();
        typename std::map< T , value_set > dint_array;
		
		
        value_set* pointer;
		
		
	
	    label =0.0;
		while (it_dint!=end_dint) {
			
			//std::cout<< "HERE" << std::endl;
            pointer=new value_set;
			//std::cout << (it_dint->second).size()<< std::endl;
			//std::cout << (it_dint->first)<< std::endl;
			
            ///(*pointer).reshape(array_shape((it_dint->second).size()));
			//std::cout<< "before the loop" << std::endl;						  
			for(int i=0; i<(it_dint->second).size();i++){
				
			
                (*pointer).push_back(*((it_dint->second)[i]));
                //(*pointer)[i]=*((it_dint->second)[i]);
			
			}
			vigra_precondition(it_dint->first>=0, "negative label encountered");
			vigra_precondition(it_dint->first==label, "Missing label!! plese relabel the image!");
			
			dint_array[it_dint->first]=*pointer;
			
			++it_dint;
			label=label+1.0;
		}
		
	
		//assign the shape to the max label. note that there are all the labels from 0 to current value of label
	
		res_shape[0]=int(label);
		
		res.reshape(res_shape);
		//std::cout<< "Passed the point" << std::endl;
		//compute the functor for each supevoxel
		typename std::map< T , three_set >::iterator it_pos=dpos.begin();
		typename std::map< T , three_set >::iterator end_pos=dpos.end();
        typename std::map< T , value_set >::iterator it_dint_array=dint_array.begin();
		
	feature_array temp;
		int i=0;	
	while (it_pos!=end_pos) {
		
		temp=(*fp)(it_pos->second,it_dint_array->second);
			
		for(int j=0; j<res_shape[1];j++) {
			i=int(it_pos->first);
			res(i,j)=temp[j];		
		}
		
		
		
		
		
		//std::cout << "HERE"<< std::endl;
		
		++it_dint_array;
		++it_pos;
		
	}
	
	
	return res;
	}

	
	
}













// DEFINITION OF THE EXPORTER


void defineOBJECTFEATURES()
{	using namespace vigra;
	using namespace python;
	
	def("extractVolume", registerConverters(&pythonExtractVolume<unsigned int, float>),
		(arg("coord"), arg("res") = python::object()),
        "Object Volume feature: \n"
		 "input structure: points set = numpy Nx3 array of numpy.uint32 \n"
		 "output structure: numpy 1x1 array of float32  \n"
		 "output[0] = volume \n");
	
	
	def("extractStatPosition", registerConverters(&pythonExtractPosition<unsigned int, float>),
		(arg("coord"), arg("res") = python::object()),
        "Object Statistics of Position feature: \n"
		"input structure: points set = numpy Nx3 array of numpy.uint32 \n"
		"output structure: numpy 12x1 array of float32  \n"
		"output[0:2] = mean x,y,z coordinate \n"
		"output[3:5] = variance of x,y,z coordinates \n"
		"output[6:8] = skew of x,y,z coordinates \n"
		"output[9:11] = kurtosis of x,y,z coordinates \n"
		"negative value is returned in case that is not possible to calsuclate the statistics");

	
	def("extractWeightedStatPosition", registerConverters(&pythonExtractWeightedPosition<unsigned int,float >),
		(arg("coord"), arg("intensities"), arg("res") = python::object()),
        "Object Weighted Statistics of Position feature: \n"
		"input structure: points set = numpy Nx3 array of numpy.uint32 \n"
		"input structure: intensities = numpy Nx1 array of numpy.float32 \n"
		"output structure: numpy 12x1 array of float32  \n"
		"output[0:2] = weighted mean x,y,z coordinate \n"
		"output[3:5] = weighted variance of x,y,z coordinates \n"
		"output[6:8] = weighted skew of x,y,z coordinates \n"
		"output[9:11] = weighted kurtosis of x,y,z coordinates \n"
		"negative value is returned in case that is not possible to calsuclate the statistics");
	
	
	def("extractPC", registerConverters(&pythonExtractPrincipalComponents<unsigned int,float >),
		(arg("coord"), arg("intensities"), arg("res") = python::object()),
        "Object Principal Components feature: \n"
		"input structure: points set = numpy Nx3 array of numpy.uint32 \n"
		"input structure: intensities = numpy Nx1 array of numpy.float32 \n"
		"output structure: numpy 12x1 array of float32  \n"
		"output[0:2] = Eigenvalues of Covariance Matrix \n"
		"output[3:5] = Normalized Eigenvector of eigenvalues [0] x,y,z coordinates centered in origin \n"
		"output[6:8] = Normalized Eigenvector of eigenvalues [1] x,y,z coordinates centered in origin \n"
		"output[9:11] = Normalized Eigenvector of eigenvalues [2] x,y,z coordinates centered in origin \n");
	
	
	
	def("extractBoundingBox", registerConverters(&pythonExtractBoundingBox<unsigned int,float >),
		(arg("coord"), arg("intensities"), arg("res") = python::object()),
        "Object Bounding Box: \n"
        "Find the smallest possible box that contains the object\n"
		"input structure: points set = numpy Nx3 array of numpy.uint32 \n"
		"input structure: intensities = numpy Nx1 array of numpy.float32 \n"
		"output structure: numpy 7x1 array of float32  \n"
		"output[0] = Object Lower x position \n"
		"output[1] = Object Lower y position \n"
		"output[2] = Object Lower z position \n"
		"output[3] = Object Upper x position \n"
		"output[4] = Object Upper y position \n"
		"output[5] = Object Upper z position \n"
        "output[6] = Object Fill Factor = <object voulume> / <size of the bounding box> \n"
		
		);
	
	
	
	def("extractStatIntensity", registerConverters(&pythonExtractStatisticsIntensity<unsigned int,float >),
		(arg("coord"), arg("intensities"), arg("res") = python::object()),
        "Object Intensity Satatistics: \n"
		"calculat eh mean intensity and its central moments\n"
		"input structure: points set = numpy Nx3 array of numpy.uint32 \n"
		"input structure: intensities = numpy Nx1 array of numpy.float32 \n"
		"output structure: numpy 4x1 array of float32  \n"
		"output[0] = Mean of intensity distribution\n"
		"output[1] = Variance of the intensity distribution \n"
		"output[2] = Skewness of the intensity distribution \n"
		"output[3] = Kurtosis of the intensity distribution \n"
		);
	
	
	
	def("extractMinMaxIntensity", registerConverters(&pythonExtractMinMaxIntensity<unsigned int,float >),
		(arg("coord"), arg("intensities"), arg("res") = python::object()),
        "Find the minimum and the maximum intensity of the the object \n"
		"and the quantiles of the intensity distribution \n"
		"input structure: points set = numpy Nx3 array of numpy.uint32 \n"
		"input structure: intensities = numpy Nx1 array of numpy.float32 \n"
		"output structure: numpy 9x1 array of float32  \n"
		"output[0] = Minimum intensity \n"
		"output[1] = Maximum intensity \n"
		"output[2] =  5% quantile \n"
		"output[3] = 10% quantile \n"
		"output[4] = 20% quantile \n"
		"output[5] = 50% quantile \n"
		"output[6] = 80% quantile \n"
		"output[7] = 90% quantile \n"
		"output[8] = 95% quantile \n"
		);
	

	def("extractMaxIntensity", registerConverters(&pythonExtractMaxIntensity<unsigned int,float >),
		(arg("coord"), arg("intensities"), arg("res") = python::object()),
        "Find the maximum intensity of the bject and its position \n"
		"input structure: points set = numpy Nx3 array of numpy.uint32 \n"
		"input structure: intensities = numpy Nx1 array of numpy.float32 \n"
		"output structure: numpy 4x1 array of float32  \n"
		"output[0] = Maximum intensity \n"
		"output[1] =  x \n"
		"output[3] =  y \n"
		"output[4] =  z \n"
		);
	
/*	
	def("extractPairwise", registerConverters(&pythonExtractPairwise<unsigned int,float >),
		(arg("coord"), arg("intensities"), arg("res") = python::object()),
		"Calculate average values of differences of neighbouring intensity values \n"
		"WARNIG: Feature not fixed yet"
		"input structure: points set = numpy Nx3 array of numpy.uint32 \n"
		"input structure: intensities = numpy Nx1 array of numpy.float32 \n"
		"output structure: numpy 4x1 array of float32  \n"
		"output[0] = Average sum over absolute distances \n"
		"output[1] = Average sum over squared distances \n"
		"output[3] = Average symmetric first derivative \n"
		"output[4] = Average second derivative \n"
		);
	
*/
	
	def("extractHOG", registerConverters(&pythonExtractHOG<unsigned int,float >),
		(arg("coord"), arg("intensities"), arg("res") = python::object()),
		"Find the Histogram of Oriented Gradients \n"
		"input structure: points set = numpy Nx3 array of numpy.uint32 \n"
		"input structure: intensities = numpy Nx1 array of numpy.float32 \n"
		"output structure: numpy 20x1 array of float32  \n"

		);
	
/*	
	def("extractSGF", registerConverters(&pythonExtractSGF<unsigned int,float >),
		(arg("coord"), arg("intensities"), arg("res") = python::object()),
		"WARNING EXPERIMENTAL \n"
		"input structure: points set = numpy Nx3 array of numpy.uint32 \n"
		"input structure: intensities = numpy Nx1 array of numpy.float32 \n"
		"output structure: numpy 4x1 array of float32  \n"
		"output[0] = Maximum intensity \n"
		"output[1] =  x \n"
		"output[3] =  y \n"
		"output[4] =  z \n"
		);
	
*/	
	
	
	//HELPERS FUNCTIONS
	
	def("dictPositions3D",registerConverters(&pythonDictObjects3D<float>),arg("image"),
        "returns a python dictionary where the key is the label of the object \n"
		"map the key with a numpy array Nx3 where N is the number of points \n"
		"and the rows columns the coordinates\n"); 
	/*
	def("dictPositions",registerConverters(&pythonDictObjects<3,float>),arg("image"),
        "returns a python dictionary where the key is the label of the object \n"
		"map the key with a numpy array Nx3 where N is the number of points \n"
		"and the rows columns the coordinates\n"); 
	
	*/

	
	
	//General extractors
	
	def("extractFeature2Dict",registerConverters(&pythonExtractSelectedFeature<float>),arg("labeled_image"),arg("image"),arg("background")="No"
		"input: labeled image as numpy.ndarray 3D of float 32\n"
		"input: intensity (the original image) as a numpy ndarray 3D of float32\n"
		"input: string feature name (see below)\n"
		"input: string if it is equals to  'Background' the label with 0 will be skipped. For any other string will be included into the result\n"
        "returns a python dictionary where the key is the label of the object \n"
		"and the associated value is the intensity of a certain feature chosen in the list:\n"
		"Volume\n"
		"Statistics Positions\n"
		"Principal Components\n"
		"Bounding Box\n"
		"Statistics Intensity\n"
		"Max Intesnisty\n"
		"MinMax Intensity\n"
		"Pairwise\n"
		"HOG\n"
		"SGF\n"
		"refer to individuls extract function for further details");
	
	
	
	def("extractFeature2NdArray",registerConverters(&pythonExtractSelectedFeature2Matrix<float>),arg("labeled_image"),arg("image"),
		"input: labeled image as numpy.ndarray 3D of float 32\n"
		"input: intensity (the original image) as a numpy ndarray 3D of float32\n"
		"input: string feature name (see below)\n"
		"input: string if it is equals to  'Background' the label with 0 will be skipped. For any other string will be included into the result\n"
        "returns a python dictionary where the key is the label of the object \n"
		"and the associated value is the intensity of a certain feature chosen in the list:\n"
		"Volume\n"
		"Statistics Positions\n"
		"Principal Components\n"
		"Bounding Box\n"
		"Statistics Intensity\n"
		"Max Intesnisty\n"
		"MinMax Intensity\n"
		"Pairwise\n"
		"HOG\n"
		"SGF\n"
		"refer to individuls extract function for further details");
}



//#include"regressionForest.hpp"

using namespace vigra;
using namespace boost::python;

BOOST_PYTHON_MODULE_INIT(objectfeatures)
{
	import_vigranumpy();
    defineOBJECTFEATURES();
    //defineRandomRegressionForest();
}



