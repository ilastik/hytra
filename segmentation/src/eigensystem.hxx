#ifndef __VIGRA_EIGENSYSTEM_HXX__
#define __VIGRA_EIGENSYSTEM_HXX__

#include <vigra/mathutil.hxx>
#include <vigra/multi_array.hxx>
#include <vigra/multi_convolution.hxx>
#include <vigra/matrix.hxx>
#include <vigra/eigensystem.hxx>
#include <vigra/array_vector.hxx>
#include <vigra/static_assert.hxx>

namespace vigra {

template <int N, int N1>
struct Tensors__Dimension_mismatch_between_input_and_output
: staticAssert::AssertBool<(N1 == N + 1)>
{};

template <int N, class T, class C1, int N1, class U, class C2>
void hessianOfGaussian(MultiArrayView<N, T, C1> const & in,
                       MultiArrayView<N1, U, C2> & hesse, double scale)
{
	const int matSize = N*(N+1)/2;

	VIGRA_STATIC_ASSERT((Tensors__Dimension_mismatch_between_input_and_output<N, N1>));

	for(int k=0; k<N; ++k)
		vigra_precondition(in.shape(k) == hesse.shape(k),
	    	"hessianOfGaussian(): Shape mismatch between input and output.");
	vigra_precondition(hesse.shape(N) == matSize,
	    "hessianOfGaussian(): Wrong number of bands in output array.");
	vigra_precondition(scale > 0.0,
	    "hessianOfGaussian(): Scale must be positive.");

    Kernel1D<double> gauss;
	gauss.initGaussian(scale);

	for(int b=0, i=0; i<N; ++i)
	{
		for(int j=i; j<N; ++j, ++b)
		{
			MultiArrayView<N, U, C2> hband = hesse.bindOuter(b);
			ArrayVector<Kernel1D<double> > kernels(N, gauss);
			if(i == j)
			{
				kernels[i].initGaussianDerivative(scale, 2);
			}
			else
			{
				kernels[i].initGaussianDerivative(scale, 1);
				kernels[j].initGaussianDerivative(scale, 1);
			}
			separableConvolveMultiArray(srcMultiArrayRange(in), destMultiArray(hband),
										kernels.begin());
		}
	}
}

template <int N, class T, class C1, int N1, class U, class C2>
void structureTensor(MultiArrayView<N, T, C1> const & in,
                     MultiArrayView<N1, U, C2> & st, double innerScale, double outerScale)
{
	const int matSize = N*(N+1)/2;

	VIGRA_STATIC_ASSERT((Tensors__Dimension_mismatch_between_input_and_output<N, N1>));

	for(int k=0; k<N; ++k)
		vigra_precondition(in.shape(k) == st.shape(k),
	    	"structureTensor(): Shape mismatch between input and output.");
	vigra_precondition(st.shape(N) == matSize,
	    "structureTensor(): Wrong number of bands in output array.");
	vigra_precondition(innerScale > 0.0 && outerScale >= 0.0,
	    "structureTensor(): Scales must be positive.");
	    
	typename MultiArrayShape<N+1>::type gradShape(st.shape(0), st.shape(1), st.shape(2), N);
	MultiArray<N+1, double> gradient(gradShape);

    Kernel1D<double> gauss;
	gauss.initGaussian(innerScale);

	for(int b=0; b<N; ++b)
	{
        MultiArrayView<N, double, UnstridedArrayTag> gband = gradient.bindOuter(b);
        ArrayVector<Kernel1D<double> > kernels(N, gauss);
        kernels[b].initGaussianDerivative(innerScale, 1);
		separableConvolveMultiArray(srcMultiArrayRange(in), destMultiArray(gband),
                                    kernels.begin());
	}

	for(int z=0; z<st.shape(2); ++z)
	{
		for(int y=0; y<st.shape(1); ++y)
		{
			for(int x=0; x<st.shape(0); ++x)
			{
				for(int b=0, i=0; i<N; ++i)
				{
					for(int j=i; j<N; ++j, ++b)
					{
						st(x, y, z, b) = gradient(x, y, z, i)*gradient(x, y, z, j);
					}
				}
			}
		}
	}
	
	for(int b=0; b<matSize; ++b)
	{
		MultiArrayView<N, U, C2> stband = st.bindOuter(b);
		
		gaussianSmoothMultiArray(srcMultiArrayRange(stband), destMultiArray(stband),
                              	 outerScale);
    }
}

template <class T1, class T2>
void eigenValuesPerVoxel(MultiArrayView<3, T1> const & vol,
                         MultiArrayView<4, T2> const & tensors,
                         MultiArrayView<4, T2> & eigenValues)
{
	const int N = 3;
	const int matSize = N*(N+1)/2;
	Matrix<double > tensor(N, N), ev(N, 1);
    eigenValues.init(0);
	for(int z=0; z<tensors.shape(2); ++z)
	{
		for(int y=0; y<tensors.shape(1); ++y)
		{
			for(int x=0; x<tensors.shape(0); ++x)
			{
				symmetric3x3Eigenvalues(tensors(x, y, z, 0), tensors(x, y, z, 1), tensors(x, y, z, 2),
					                    tensors(x, y, z, 3), tensors(x, y, z, 4), tensors(x, y, z, 5),
										&eigenValues(x, y, z, 0), &eigenValues(x, y, z, 1), &eigenValues(x, y, z, 2));
			}
		}
	}
}


template <class T1, class T2>
void eigenValuesPerPixel(MultiArrayView<2, T1> const & vol,
                         MultiArrayView<3, T2> const & tensors,
                         MultiArrayView<3, T2> & eigenValues)
{
	const int N = 2;
	const int matSize = N*(N+1)/2;
	Matrix<double > a(N, N), ew(N, 1), ev(N, N);
    eigenValues.init(0);
    for(int y=0; y<tensors.shape(1); ++y) {
        for(int x=0; x<tensors.shape(0); ++x) {
            a(0, 0) = tensors(x, y, 0);
            a(0, 1) = tensors(x, y, 1);
            a(1, 0) = tensors(x, y, 1);
            a(1, 1) = tensors(x, y, 2);

            symmetricEigensystem(a, ew, ev);

            eigenValues(x, y, 0) = ew[0];
            eigenValues(x, y, 1) = ew[1];
        }
    }
}


} // namespace vigra

#endif // __VIGRA_EIGENSYSTEM_HXX__
