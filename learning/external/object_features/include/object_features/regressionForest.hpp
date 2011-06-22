/*
 *  regressionForest.cpp
 *  vigra
 *
 *  Created by Luca Fiaschi on 3/14/11.
 *  Copyright 2011 Heidelberg university. All rights reserved.
 *
 */


#define PY_ARRAY_UNIQUE_SYMBOL vigranumpylearning_PyArray_API
// #define NO_IMPORT_ARRAY

#include <vigra/numpy_array.hxx>
#include <vigra/numpy_array_converters.hxx>
#include <vigra/random_forest.hxx>
#include <set>
#include <cmath>
#include <memory>
#include <boost/python.hpp>

namespace python = boost::python;
namespace vigra
{
	

	
	template<class LabelType, class FeatureType>
	RandomForest<LabelType,RegressionTag>*
	pythonConstructRegressionRandomForest(int treeCount)
	
	
	{
				
		RandomForest<LabelType,RegressionTag>* rf = new RandomForest<LabelType,RegressionTag>(vigra::RandomForestOptions().tree_count(treeCount));
		
		return rf;
	}
	

	template<class LabelType, class FeatureType>
	void
	pythonLearnRegressionRandomForest(RandomForest<LabelType,RegressionTag> & rf, 
							NumpyArray<2,FeatureType> trainData, 
							NumpyArray<2,LabelType> trainLabels)
	{
		vigra::RegressionSplit rsplit;
		
		
			
			rf.learn(trainData, trainLabels,rf_default(),rsplit);
	
	
	}
	
	
	template<class LabelType,class FeatureType>
	NumpyAnyArray 
	pythonRFRegressionPredict(RandomForest<LabelType,RegressionTag> const & rf,
						  NumpyArray<2,FeatureType> testData,
							  NumpyArray<2,LabelType> res=python::object())
	{
		//construct result
		res.reshapeIfEmpty(MultiArrayShape<2>::type(testData.shape(0),1),
						   "Output array has wrong dimensions.");
		rf.predictRaw(testData,res);
		return res;
	}
	

	void defineRandomRegressionForest()
	{
		using namespace python;
		
		docstring_options doc_options(true, true, false);
		
		
		class_<RandomForest<float,RegressionTag> > rfclass_new("RandomRegressionForest",python::no_init);
		
		rfclass_new
        .def("__init__",python::make_constructor(registerConverters(&pythonConstructRegressionRandomForest<float,float>),
                                                 boost::python::default_call_policies(),
                                                 ( arg("treeCount")=255)),
												  
             "Constructor::\n\n"
             "  RandomForest(treeCount = 255, mtry=RF_SQRT, min_split_node_size=1,\n"
             "               training_set_size=0, training_set_proportions=1.0,\n"
             "               sample_with_replacement=True, sample_classes_individually=False,\n"
             "               prepare_online_learning=False)\n\n"
             "'treeCount' controls the number of trees that are created.\n\n"
             "See RandomForest_ and RandomForestOptions_ in the C++ documentation "
												  "for the meaning of the other parameters.\n")
        //.def("featureCount",
		//	 &RandomForest<UInt32>::column_count,
        //     "Returns the number of features the RandomForest works with.\n")
        //.def("labelCount",
		//	 &RandomForest<UInt32>::class_count,
        //     "Returns the number of labels, the RandomForest knows.\n")
        //.def("treeCount",
        //     &RandomForest<UInt32>::tree_count,
        //     "Returns the 'treeCount', that was set when constructing the RandomForest.\n")
        .def("predict",
             registerConverters(&pythonRFRegressionPredict<float,float>),
             (arg("testData"),arg("res")=python::object()),
             "Predict labels on 'testData'.\n\n"
             "The output is an array containing a labels for every test samples.\n")
        .def("learnRF",
             registerConverters(&pythonLearnRegressionRandomForest<float,float>),
             (arg("trainData"), arg("trainLabels")),
             "Trains a random Forest using 'trainData' and 'trainLabels'.\n\n"
             "and returns the OOB. See the vigra documentation for the meaning af the rest of the paremeters.\n")
												 ;
	}
	
} // namespace vigra

/*
using namespace vigra;
using namespace boost::python;
												 
BOOST_PYTHON_MODULE_INIT(objectfeatures)
{
    import_vigranumpy();
    defineRandomRegressionForest();

}

*/
