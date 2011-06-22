#ifndef __VIGRA_ALGORITHM_PACKAGE__
#define __VIGRA_ALGORITHM_PACKAGE__

#include <iostream>
#include <vigra/matrix.hxx>
#include <vigra/multi_convolution.hxx>
#include <vigra/multi_resize.hxx>
#include "eigensystem.hxx"

using namespace vigra;

template <int DIM, class TIN, class TOUT >
void vigraIntensityThresholding(
    const MultiArrayView<DIM, TIN > &vol,
    const TIN threshold,
    MultiArray<DIM, TOUT > &filtered) 
{
	filtered.reshape(vol.shape());
    filtered.copy(vol);
    for (int i=0; i<vol.elementCount(); i++) {
        //filtered[i] = vol[i] > threshold ? static_cast<TOUT >(round(vol[i])) : 0;
        if (filtered[i] < threshold)
            filtered[i] = 0;
    }
}

template <int DIM, class TIN, class TOUT >
void vigraResize(
    const MultiArrayView<DIM, TIN > &vol,
    const typename MultiArrayView<DIM, TIN >::difference_type &interpolation,
    MultiArray<DIM, TOUT > &resizedImage) 
{
	// shapes
	typedef typename MultiArrayView<DIM, TIN >::difference_type Shape;

	// outputs
    Shape shapeInterp;
    for (int i=0; i<DIM; i++)
        shapeInterp[i] = vol.shape(i)*interpolation[i];
	resizedImage.reshape(shapeInterp);

	// resizing
	BSpline<DIM, TOUT > splines;
    resizeMultiArraySplineInterpolation(
		srcMultiArrayRange(vol), 
		destMultiArrayRange(resizedImage),
		splines);
}

template <class T>
void vigraGaussianSmoothing3(const MultiArrayView<3, T > &vol, 
                            const float sigma,
							MultiArrayView<3, T > smoothedImage) 
{
	// shapes
	typedef typename MultiArrayView<4, T>::difference_type Shape4;
	typedef typename MultiArrayView<3, T>::difference_type Shape3;
	Shape3 shape(vol.shape());

	// smoothing over region at the given scales
	ArrayVector<Kernel1D<T > > kernels(3);
	for ( unsigned int k = 0; k < 3; ++k ) {
		Kernel1D<T > kernel;
		kernel.initGaussian(sigma);
		kernels[k] = kernel;
	}

	// smooth using the narrow kernel
	separableConvolveMultiArray(srcMultiArrayRange(vol), 
		destMultiArray(smoothedImage), kernels.begin());
}

template <class T>
void vigraGaussianSmoothing3(const MultiArrayView<3, T > &vol, 
                            const Matrix<T > &sigmas,
							MultiArray<3, T > &smoothedImage) 
{
	// shapes
	typedef typename MultiArrayView<4, T>::difference_type Shape4;
	typedef typename MultiArrayView<3, T>::difference_type Shape3;
	Shape3 shape(vol.shape());

	// smoothing over region at the given scales
	ArrayVector<Kernel1D<T > > kernels(3);
	for ( unsigned int k = 0; k < 3; ++k ) {
		Kernel1D<T > kernel;
		kernel.initGaussian(sigmas[k]);
		kernels[k] = kernel;
	}

	// smooth using the narrow kernel
	separableConvolveMultiArray(srcMultiArrayRange(vol), 
		destMultiArray(smoothedImage), kernels.begin());
}

template <class T>
void vigraGaussianGradientMagnitude3(const MultiArrayView<3, T> &vol,
									 const float sigma,
									 MultiArrayView<3, T> gradMag) 
{	// shapes
	typedef typename MultiArrayView<4, T>::difference_type Shape4;
	typedef typename MultiArrayView<3, T>::difference_type Shape3;
	Shape3 shape(vol.shape());
    
    // initialize outputs
    gradMag.init(0);
    
    // compute gradient components
    for(unsigned int d = 0; d < 3; ++d ) {
        MultiArray<3, T > partialGrad(shape);
        ArrayVector<Kernel1D<T > > kernels(3);
        for (unsigned int k = 0; k < 3; ++k ) {
            Kernel1D<T > kernel;
            if (k == d)
                kernel.initGaussianDerivative(sigma, 1);
            else
                kernel.initGaussian(sigma);
            kernels[k] = kernel;
        }
        separableConvolveMultiArray(srcMultiArrayRange(vol), destMultiArray(partialGrad), kernels.begin());
        combineTwoMultiArrays(srcMultiArrayRange(partialGrad), srcMultiArray(partialGrad), 
                destMultiArray(partialGrad), std::multiplies<T >());
        gradMag += partialGrad;
    }
    vigra::transformMultiArray(srcMultiArrayRange(gradMag), 
            destMultiArray(gradMag), (float(*)(float))&std::sqrt);
}

template <class T>
void vigraDifferenceOfGaussian3(const MultiArrayView<3, T> &vol,
								const float sigma1, const float sigma2,
								MultiArrayView<3, T> diffOfGaussian) 
{
	// shapes
	typedef typename MultiArrayView<4, T>::difference_type Shape4;
	typedef typename MultiArrayView<3, T>::difference_type Shape3;
	Shape3 shape(vol.shape());

	MultiArray<3, T > smoothedImage(shape);

	// smoothing over region at the given scales
	ArrayVector<Kernel1D<T > > kernels1(3);
	ArrayVector<Kernel1D<T > > kernels2(3);
	for ( unsigned int k = 0; k < 3; ++k ) {
		Kernel1D<T > kernel1;
		kernel1.initGaussian(sigma1);
		kernels1[k] = kernel1;

		Kernel1D<T > kernel2;
		kernel2.initGaussian(sigma2);
		kernels2[k] = kernel2;
	}

	// smooth using the narrow kernel
	separableConvolveMultiArray(srcMultiArrayRange(vol), 
		destMultiArray(diffOfGaussian), kernels1.begin());

	// smooth using the wide kernel
	separableConvolveMultiArray(srcMultiArrayRange(vol), 
		destMultiArray(smoothedImage), kernels2.begin());
	diffOfGaussian -= smoothedImage;
}

template <class T>
void vigraEigenValueOfHessianMatrix3(const MultiArrayView<3, T> &vol,
									 const float sigma,
									 MultiArrayView<4, T> eigenValues) 
{
	// shapes
	typedef typename MultiArrayView<4, T>::difference_type Shape4;
	typedef typename MultiArrayView<3, T>::difference_type Shape3;
	Shape3 shape(vol.shape());

	// compute the partial 2nd derivatives: d1, d2 are the index of the first/second variables, i.e. dimensions
	MultiArray<4, T > hessianVector(Shape4(vol.shape(0), vol.shape(1), vol.shape(2), 6));
	unsigned int h = 0;     // h is the index of the hessian matrix elements
	ArrayVector<Kernel1D<T > > kernels(3);     // kernels
	for ( unsigned int d1=0; d1 < 3; ++d1 ) {
		for ( int d2 = d1; d2 < 3; ++d2 ) {
			MultiArrayView<3, T> hessianVectorElement = hessianVector.bindOuter(h++);
			ArrayVector<Kernel1D<T > > kernels(3);
			for (unsigned int k = 0; k < 3; ++k ) {
				Kernel1D<T > kernel;
				if (k == d1 && k == d2)
					kernel.initGaussianDerivative(sigma, 2);
				else if (k == d1 || k == d2)
					kernel.initGaussianDerivative(sigma, 1);
				else
					kernel.initGaussian(sigma);
				kernels[k] = kernel;
			}
			separableConvolveMultiArray(srcMultiArrayRange(vol), destMultiArray(hessianVectorElement),
				kernels.begin());
		}
	}

	// compute the eigen values
	eigenValuesPerVoxel<T, T >(vol, hessianVector, eigenValues);
}

template <class T>
void vigraEigenValueOfStructureTensor3(const MultiArrayView<3, T> &vol,
									const float sigma, 
                                    const float scale,
									MultiArrayView<4, T> eigenValues)

{
	// shapes
	typedef typename MultiArrayView<4, T>::difference_type Shape4;
	typedef typename MultiArrayView<3, T>::difference_type Shape3;
	Shape3 shape(vol.shape());

    // settings
    typename MultiArray<4, T >::difference_type gradientShape(vol.shape(0), vol.shape(1), vol.shape(2), 3);
    typename MultiArray<4, T >::difference_type eigenValuesShape(vol.shape(0), vol.shape(1), vol.shape(2), 3);
    typename MultiArray<4, T >::difference_type structurelTensorVectorShape(vol.shape(0), vol.shape(1), vol.shape(2), 6);
 	MultiArray<4, T> gradient(gradientShape);
    MultiArray<4, T> structurelTensor(structurelTensorVectorShape);
    
    // compute gradient components
    for (unsigned int d = 0; d < 3; ++d ) {
        MultiArrayView<3, T > partialGrad = gradient.bindOuter(d);
        ArrayVector<Kernel1D<T > > kernels(3);
        for (unsigned int k = 0; k < 3; ++k ) {
            Kernel1D<T > kernel;
            if (k == d)
                kernel.initGaussianDerivative(sigma, 1);
            else
                kernel.initGaussian(sigma);
            kernels[k] = kernel;
        }
        separableConvolveMultiArray(srcMultiArrayRange(vol), destMultiArray(partialGrad), kernels.begin());
    }
    
    // compute structure tensor
    for(unsigned int z=0; z<structurelTensor.shape(2); ++z)
        for(unsigned int y=0; y<structurelTensor.shape(1); ++y)
            for(unsigned int x=0; x<structurelTensor.shape(0); ++x)
                for(unsigned int b=0, i=0; i<3; ++i)
                    for(unsigned int j=i; j<3; ++j, ++b)
                        structurelTensor(x, y, z, b) = gradient(x, y, z, i)*gradient(x, y, z, j);
    
    // smoothing over region at the given scale
    if (scale > 0) {
        ArrayVector<Kernel1D<T > > kernels(3);
        for ( unsigned int k = 0; k < 3; ++k ) {
            Kernel1D<T > kernel;
            kernel.initGaussian(scale);
            kernels[k] = kernel;
        }
        for ( unsigned int st = 0; st < structurelTensor.shape(3); ++st ) {
            MultiArrayView<3, T> structurelTensorElement = structurelTensor.bindOuter(st);
            separableConvolveMultiArray(srcMultiArrayRange(structurelTensorElement), 
                    destMultiArray(structurelTensorElement), kernels.begin());
        }
    }
    
    // compute the eigen values
    eigenValuesPerVoxel<T, T >(vol, structurelTensor, eigenValues);
}


template <class T > void vigraGaussianGradient3(
        const MultiArrayView<3, T > &data,
        const float sigma, 
        MultiArray<4, T > &gvf)
{
    typedef typename MultiArray<4, T >::difference_type ShapeGVF;
    
    gvf.reshape(ShapeGVF(data.shape(0), data.shape(1), data.shape(2), 3), 0);
        
    /* compute the gradient */
    for(unsigned int d = 0; d < 3; ++d ) {
        MultiArrayView<3, T > gradPartial = gvf.bindOuter(d);
        ArrayVector<Kernel1D<T > > kernels(3);
        for (unsigned int k = 0; k < 3; ++k ) {
            Kernel1D<T > kernel;
            if (k == d)
                kernel.initGaussianDerivative(sigma, 1);
            else
                kernel.initGaussian(sigma);
            kernels[k] = kernel;
        }
        separableConvolveMultiArray(srcMultiArrayRange(data), destMultiArray(gradPartial), kernels.begin());
    }
}

#endif /* __VIGRA_ALGORITHM_PACKAGE__ */
