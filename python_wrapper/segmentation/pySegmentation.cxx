#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <vigra/numpy_array.hxx>
#include <vigra/numpy_array_converters.hxx>
#include <vigra/multi_array.hxx>

#include "../../segmentation/examples/segmentation.cxx"
#include "../../segmentation/include/ctSegmentationMSA.hxx"
#include "../../segmentation/include/ctSegmentationRoutine.hxx"



namespace ctSegmentationMSAext {
    template<int DIM, class TIN, class TOUT>
    vigra::NumpyArray<DIM, TOUT, vigra::UnstridedArrayTag> 
    run(ctSegmentationMSA& r, const vigra::NumpyArray<DIM, TIN, vigra::UnstridedArrayTag>& data){
       vigra::NumpyArray<DIM, TOUT, vigra::UnstridedArrayTag> result(data.shape());
       r.run<DIM, TIN, TOUT>(data, result);
       return result;
    }
}


BOOST_PYTHON_MODULE( segmentation )
{
    using namespace boost::python;
 
    // helper
    class_<std::vector<float> >("FloatVec")
        .def(vector_indexing_suite<std::vector<float> >())
    ;

    // segmentation.cxx
    def("doSegmentation", doSegmentation);

    // ctSegmentationRoutine.hxx
    def("split_export", split_export<3, unsigned short, unsigned short> );
    def("interpolate_segment_filter", interpolate_segment_filter<3, unsigned short, unsigned short> );

    // ctSegmentationMSA.hxx
    vigra::import_vigranumpy(); 
    class_<ctSegmentationMSA>("ctSegmentationMSA", init<std::vector<float>, int, int, std::vector<float> >())
	.def("printParameters", &ctSegmentationMSA::print)
	.def("run", vigra::registerConverters(&ctSegmentationMSAext::run<3, float, unsigned short>), (arg("segmenter"), arg("volume")))
    ;

}
