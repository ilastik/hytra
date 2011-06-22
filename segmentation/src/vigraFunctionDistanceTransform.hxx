#ifndef __FUNCTION_DISTANCE_TRANSFORM__
#define __FUNCTION_DISTANCE_TRANSFORM__

#include <vigra/symmetry.hxx>

using namespace vigra;

#define INF     1e20
#define C       StridedArrayTag

/* dt of 1d function using squared distance */
template <class T> void functionDistanceTransform(
        const MultiArrayView<1, T, C > &f,
        MultiArrayView<1, T, C > d) 
{   
    typedef typename MultiArrayView<1, T, C >::difference_type Shape1D;
    int n = f.shape(0);
        
    MultiArray<1, int > v(f.shape());
    MultiArray<1, float > z(f.shape()+Shape1D(1));
    int k = 0;
    v[0] = 0;
    z[0] = -INF;
    z[1] = +INF;
    for (int q = 1; q <= n-1; q++) {
        float s  = ((f[q]+q*q)-(f[v[k]]+v[k]*v[k]))/(2*q-2*v[k]);
        while (s <= z[k]) {
            k--;
            s  = ((f[q]+q*q)-(f[v[k]]+v[k]*v[k]))/(2*q-2*v[k]);
        }
        k++;
        v[k] = q;
        z[k] = s;
        z[k+1] = +INF;
    }
    
    k = 0;
    for (int q = 0; q <= n-1; q++) {
        while (z[k+1] < q)
            k++;
        d[q] = (q-v[k])*(q-v[k]) + f[v[k]];
    }
}

/* dt of 2d function using squared distance */
template <class T > void functionDistanceTransform(
        const MultiArrayView<2, T, C > &f,
        MultiArrayView<2, T, C > d) 
{    
    // transform along columns
    MultiArray<2, T > tmpMultiArray(f.shape());
    MultiArrayView<2, T, C > tmp(tmpMultiArray.shape(), tmpMultiArray.data());
    for (int x = 0; x < f.shape(0); x++) {
        MultiArrayView<1, T, C > colF = f.bindInner(x);
        MultiArrayView<1, T, C > colTmp = tmp.bindInner(x);
        functionDistanceTransform<T >(colF, colTmp);
    }
    
    // transform along rows
    for (int y = 0; y < f.shape(1); y++) {
        MultiArrayView<1, T, C > rowF = tmp.bindOuter(y);
        MultiArrayView<1, T, C > rowD = d.bindOuter(y);
        functionDistanceTransform<T >(rowF, rowD);
    }
}

/* dt of 3d function using squared distance */
template <class T > void functionDistanceTransform(
        const MultiArrayView<3, T, C > &f,
        MultiArrayView<3, T, C > d) 
{
    // transform each slice along the z-axis
    MultiArray<3, T > tmpMultiArray(f.shape());
    MultiArrayView<3, T, C > tmp(tmpMultiArray.shape(), tmpMultiArray.data());
    for (int z = 0; z < f.shape(2); z++) {
        MultiArrayView<2, T, C > colF = f.bindOuter(z);
        MultiArrayView<2, T, C > colTmp = tmp.bindOuter(z);
        functionDistanceTransform<T >(colF, colTmp);
    }
    
    // transform along the rows and columns
    for (int x = 0; x < f.shape(0); x++) {
        for (int y = 0; y < f.shape(1); y++) {
            //MultiArrayView<1, T, C > lineF = tmp.bindInner(TinyVector<int, 2>(x, y));
            MultiArrayView<2, T, C > planeF = tmp.bindInner(x);
            MultiArrayView<1, T, C > lineF = planeF.bindInner(y);
            //MultiArrayView<1, T, C > lineD = d.bindInner(TinyVector<int, 2>(x, y));
            MultiArrayView<2, T, C > planeD = d.bindInner(x);
            MultiArrayView<1, T, C > lineD = planeD.bindInner(y);
            functionDistanceTransform<T >(lineF, lineD);
        }
    }
}

#endif /* __FUNCTION_DISTANCE_TRANSFORM__ */