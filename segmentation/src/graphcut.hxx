#ifndef __GRAPHCUT_HXX__
#define __GRAPHCUT_HXX__

#include <iostream>
#include <algorithm>

// max-flow library
//#include "graph.h"
#include <graph.cpp>
#include <maxflow.cpp>

using namespace vigra;

// t-link method definition
#define TLINK_METHOD_PROBMAP      0x0001
#define TLINK_METHOD_FLUX         0x0002
#define TLINK_METHOD_NORMALIZED   0x0004
#define TLINK_METHOD_LOGISTIC     0x0008
#define TLINK_METHOD_LOGLOGISTIC  0x0010
#define TLINK_METHOD_BOUNDARYCUE  0x0020
#define TLINK_METHOD_SEEDED       0x0040

// n-link method definition
#define NLINK_METHOD_GAUSSIAN     0x0001
#define NLINK_METHOD_INVERSE      0x0002
#define NLINK_METHOD_SHAPE        0x0004

// get neighbors
template <int NDIM > std::vector<typename MultiArrayView<NDIM, double >::difference_type > 
        createNeighborhood(const int& neighborhood = 4, bool all = false) 
{
    typedef typename MultiArrayView<NDIM, double >::difference_type Shape;

    std::vector<Shape > neighbors;
    
    if (NDIM == 2) {
        // by default, add the 4 neighborhood, i.e. geodesic distance = 1
        neighbors.push_back(Shape(+1,  0));
        neighbors.push_back(Shape( 0, +1));
        if (all) {
            neighbors.push_back(Shape(-1,  0));
            neighbors.push_back(Shape( 0, -1));
        }

        // if necessary, add those with geodesic distance = 2
        if (neighborhood == 8) {
            neighbors.push_back(Shape(-1, +1));
            neighbors.push_back(Shape(+1, +1));
            if (all) {
                neighbors.push_back(Shape(-1, -1));
                neighbors.push_back(Shape(+1, -1));
            }
        }
    }
    else {
        // by default, add the 6 neighborhood, i.e. geodesic distance = 1
        neighbors.push_back(Shape(+1,  0,  0));
        neighbors.push_back(Shape( 0, +1,  0));
        neighbors.push_back(Shape( 0,  0, +1));
        if (all) {
            neighbors.push_back(Shape(-1,  0,  0));
            neighbors.push_back(Shape( 0, -1,  0));
            neighbors.push_back(Shape( 0,  0, -1));
        }

        // if necessary, add those with geodesic distance = 2 or 3
        if (neighborhood == 26) {
            neighbors.push_back(Shape(+1, +1,  0));
            neighbors.push_back(Shape(-1, +1,  0));
            neighbors.push_back(Shape(+1,  0, +1));
            neighbors.push_back(Shape(-1,  0, +1));
            neighbors.push_back(Shape( 0, +1, +1));
            neighbors.push_back(Shape( 0, -1, +1));
            
            neighbors.push_back(Shape(+1, +1, +1));
            neighbors.push_back(Shape(-1, +1, +1));
            neighbors.push_back(Shape(+1, -1, +1));
            neighbors.push_back(Shape(-1, -1, +1));
            
            if (all) {
                neighbors.push_back(Shape(+1, -1,  0));
                neighbors.push_back(Shape(-1, -1,  0));
                neighbors.push_back(Shape(+1,  0, -1));
                neighbors.push_back(Shape(-1,  0, -1));
                neighbors.push_back(Shape( 0, +1, -1));
                neighbors.push_back(Shape( 0, -1, -1));

                neighbors.push_back(Shape(+1, +1, -1));
                neighbors.push_back(Shape(-1, +1, -1));
                neighbors.push_back(Shape(+1, -1, -1));
                neighbors.push_back(Shape(-1, -1, -1));
            }
        }
    }
    
    return neighbors;
}

// estimate the number of edges: 2D
template <int NDIM > unsigned int estimateNumberOfEdges(
        const typename MultiArrayView<NDIM, double >::difference_type& shape, 
        const int& neighborhood = 4) 
{
    int nEdges = 0;

    if (NDIM == 2) {
        // by default, use the 4 neighborhood, i.e. geodesic distance = 1
        nEdges += 2 * shape[0] * shape[1] - shape[0] - shape[1];

        // if necessary, add those with geodesic distance = 2
        if (neighborhood == 8) {
            nEdges += 2*(shape[0] - 1)*(shape[1] - 1);
        }
    }
    else {
        // by default, use the 6 neighborhood, i.e. geodesic distance = 1
        nEdges += 3*shape[0]*shape[1]*shape[2]
                - shape[0]*shape[1]
                - shape[0]*shape[2]
                - shape[1]*shape[2];

        // if necessary, add those with geodesic distance = 2 or 3
        if (neighborhood == 26) {
            nEdges += 6*shape[0]*shape[1]*shape[2]
                - 4*(shape[0]*shape[1]+shape[0]*shape[2]+shape[1]*shape[2])
                + 2*(shape[0]+shape[1]+shape[2]);
            
            nEdges += 4*(shape[0]-1)*(shape[1]-1)*(shape[2]-1);
        }
    }
    
    return nEdges;
}

// compute N-link cost: gaussian
template <class TIN, class TOUT > 
        TOUT costNLinkGaussian(TIN Ip, TIN Iq, double sigma)
{
    TOUT Idelta = Ip - Iq;
    return exp(-static_cast<TOUT >(Idelta*Idelta/(sigma*sigma)));
}

// compute N-link cost: inverse
template <class TIN, class TOUT > 
        TOUT costNLinkInverse(TIN Ip, TIN Iq)
{
    TOUT Idelta = Ip - Iq;
    return 1 / fabs(static_cast<TOUT >(Idelta));
}

// compute N-link cost: shape
template <int NDIM, class TIN, class TOUT > 
    TOUT costNLinkShape(const typename MultiArrayView<NDIM, double >::difference_type &p,
                        const typename MultiArrayView<NDIM, double >::difference_type &q, 
                        const MultiArrayView<1, TIN, StridedArrayTag > &v,
                        double alpha)
{
    TOUT vDotProduct = 0;
    TOUT magEdge = 0;
    for (int i=0; i<NDIM; i++) {
        TOUT vDiff = q[i] - p[i];
        vDotProduct += vDiff * v[i];
        magEdge += vDiff * vDiff;
    }
    
    vDotProduct /= sqrt(magEdge);
    
//    if (vDotProduct >= alpha)
  //      vDotProduct = 1000;
//    else
  //      vDotProduct = -1;
    
//    return exp(-vDotProduct);

    if (vDotProduct >= alpha) 
        return 0;

    return vDotProduct * vDotProduct;
    
//    return 1/(alpha*vDotProduct));
}

// n-link method settings
int getNLinkMethodCode(const std::string &method)
{
    int codeMethod = 0;
    
    // gaussian
    if (method.find("gaussian") != std::string::npos) 
        codeMethod = codeMethod | NLINK_METHOD_GAUSSIAN;
    
    // inverse
    if (method.find("inverse") != std::string::npos) 
        codeMethod = codeMethod | NLINK_METHOD_INVERSE;
    
    // vector field
    if (method.find("shape") != std::string::npos) 
        codeMethod = codeMethod | NLINK_METHOD_SHAPE;
    
    return codeMethod;
}

// t-link method settings
int getTLinkMethodCode(const std::string &method)
{
    int codeMethod = 0;
    
    // normalized
    if (method.find("normalized") != std::string::npos) 
        codeMethod = codeMethod | TLINK_METHOD_NORMALIZED;
    
    // flux
    if (method.find("flux") != std::string::npos) 
        codeMethod = codeMethod | TLINK_METHOD_FLUX;
    
    // probability map
    if (method.find("probmap") != std::string::npos) 
        codeMethod = codeMethod | TLINK_METHOD_PROBMAP;
    
    // logistic
    if (method.find("logistic") != std::string::npos) 
        codeMethod = codeMethod | TLINK_METHOD_LOGISTIC;
    
    // log-logistic
    if (method.find("log-logistic") != std::string::npos) 
        codeMethod = codeMethod | TLINK_METHOD_LOGLOGISTIC;
    
    // boundary cue
    if (method.find("bdcue") != std::string::npos) 
        codeMethod = codeMethod | TLINK_METHOD_BOUNDARYCUE;
    
    // seeded
    if (method.find("seeded") != std::string::npos) 
        codeMethod = codeMethod | TLINK_METHOD_SEEDED;

    return codeMethod;
}

// compute T-link cost: flux
template <int NDIM, class TIN, class TOUT > TOUT costTLinkFlux(
        const typename MultiArrayView<NDIM, double >::difference_type &p,
        const typename MultiArrayView<NDIM, double >::difference_type &q, 
        const MultiArrayView<1, TIN, StridedArrayTag > &v)
{
    TOUT vDotProduct = 0;
    TOUT magEdge = 0;
    for (int i = 0; i < NDIM; i ++) {
        TOUT vDiff = q[i] - p[i];
        vDotProduct += vDiff * v[i];
        magEdge += vDiff * vDiff;
    }

    vDotProduct /= sqrt(magEdge);
    
    return vDotProduct;
}

// compute T-link cost: logistic
template <class TIN, class TOUT > TOUT costTLinkLogistic(
        const TIN &Ip, double a, double b)
{
    TOUT v = 1 / (1+exp(static_cast<TOUT >(b*(Ip - a))));
    
    return v;
}

#endif /* __GRAPHCUT_HXX__ */