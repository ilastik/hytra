#ifndef __CT_SEGMENTATION_GC__
#define __CT_SEGMENTATION_GC__

#include <iostream>
#include <vector>
#include <ctime>
#include <vigra/multi_array.hxx>
#include "ctIniConfiguration.hxx"
#include "graphcut.hxx"
    
using namespace vigra;

class ctSegmentationGC
{
public:
    // constructor: load parameters from the ini file
    ctSegmentationGC(const CSimpleIniA &ini, int verbose = false) : verbose(verbose)
    {
        // prefix
        prefix = "ctSegmentationGC: ";

        // parameters        
        neighborhood = atoi(ini.GetValue(INI_SECTION_GRAPH_CUT, "neighborhood", "6"));
        typeTLink = atoi(ini.GetValue(INI_SECTION_GRAPH_CUT, "type_tlink", "99"));
        typeNLink = atoi(ini.GetValue(INI_SECTION_GRAPH_CUT, "type_nlink", "5"));
        
        lambdaFlux = atof(ini.GetValue(INI_SECTION_GRAPH_CUT, "lambda_flux", "1"));
        lambdaProbMap = atof(ini.GetValue(INI_SECTION_GRAPH_CUT, "lambda_probmap", "1"));
        lambdaBoundaryCue = atof(ini.GetValue(INI_SECTION_GRAPH_CUT, "lambda_bdcue", "1"));
        lambdaGaussian = atof(ini.GetValue(INI_SECTION_GRAPH_CUT, "lambda_gaussian", "1"));
        lambdaShape = atof(ini.GetValue(INI_SECTION_GRAPH_CUT, "lambda_shape", "1"));
        sigmaNoise = atof(ini.GetValue(INI_SECTION_GRAPH_CUT, "sigma_noise", "25"));
        
        // verbose ?
//        verbose = atoi(ini.GetValue(INI_SECTION_RUNTIME, "verbose", "0"));
        
        // print parameters
        if (verbose)
            print();
    }
    
    void print() 
    {
        std::cout << prefix << "parameters ->" << std::endl;
        std::cout << "\t\t\t\t neighborhood = " << neighborhood << std::endl;
        std::cout << "\t\t\t\t typeTLink = " << typeTLink << std::endl;
        std::cout << "\t\t\t\t\t\t typeTLink & TLINK_METHOD_PROBMAP = " << (typeTLink & TLINK_METHOD_PROBMAP != 0) << std::endl;
        std::cout << "\t\t\t\t\t\t typeTLink & TLINK_METHOD_FLUX = " << (typeTLink & TLINK_METHOD_FLUX != 0) << std::endl;
        std::cout << "\t\t\t\t\t\t typeTLink & TLINK_METHOD_BOUNDARYCUE = " << (typeTLink & TLINK_METHOD_BOUNDARYCUE != 0) << std::endl;
        std::cout << "\t\t\t\t\t\t typeTLink & TLINK_METHOD_SEEDED = " << (typeTLink & TLINK_METHOD_SEEDED != 0) << std::endl;
        std::cout << "\t\t\t\t typeNLink = " << typeNLink << std::endl;
        std::cout << "\t\t\t\t\t\t typeNLink & NLINK_METHOD_GAUSSIAN = " << (typeNLink & NLINK_METHOD_GAUSSIAN != 0) << std::endl;
        std::cout << "\t\t\t\t\t\t typeNLink & NLINK_METHOD_SHAPE = " << (typeNLink & NLINK_METHOD_SHAPE != 0) << std::endl;
        std::cout << "\t\t\t\t lambdaFlux = " << lambdaFlux << std::endl;
        std::cout << "\t\t\t\t lambdaProbMap = " << lambdaProbMap << std::endl;
        std::cout << "\t\t\t\t lambdaBoundaryCue = " << lambdaBoundaryCue << std::endl;
        std::cout << "\t\t\t\t lambdaGaussian = " << lambdaGaussian << std::endl;
        std::cout << "\t\t\t\t lambdaShape = " << lambdaShape << std::endl;
        std::cout << "\t\t\t\t sigmaNoise = " << sigmaNoise << std::endl;
    }

    // create the graph
    template <int DIM, class FLOAT > Graph<FLOAT, FLOAT, FLOAT >* 
            createGraph(MultiArray<DIM, FLOAT > &data) 
    {
        if (verbose)
            std::cerr << prefix << "create graph object" << std::endl;
        // construct graph        
        int nNodes = data.elementCount();
        unsigned int nEdges = estimateNumberOfEdges<DIM >(data.shape(), neighborhood);

        // initialize graph
        return (new Graph<FLOAT, FLOAT, FLOAT >(nNodes, nEdges));
    }

    // call the max-flow algorithm
    template <int DIM, class FLOAT, class INT > void callMaxFlow(
            Graph<FLOAT, FLOAT, FLOAT >* g, 
            MultiArray<DIM, INT > &seg) 
    {
        if (verbose)
            std::cerr << prefix << "call the max-flow algorithm" << std::endl;
        
        FLOAT flow = g->maxflow();
        for (int i = 0; i < seg.elementCount(); i++)
            seg[i] = g->what_segment(i);
    }

    // add t-link costs
    template <int DIM, class INT, class FLOAT > void initializeTLinkCosts(
            Graph<FLOAT, FLOAT, FLOAT > *g, 
            MultiArray<DIM, FLOAT > &data, 
            MultiArray<DIM, INT > &mask, 
            MultiArray<DIM, FLOAT > &probmap, 
            MultiArray<DIM+1, FLOAT > &gvf, 
            MultiArray<DIM, FLOAT > &bdcue) 
    {
        typedef typename MultiArrayView<DIM, double >::difference_type Shape;
        typedef MultiArrayView<2, double >::difference_type Shape2D;

        if (verbose)
            std::cerr << prefix << "initialize t-link costs ..." << std::endl;

        // add nodes
        int nNodes = data.elementCount();
        g->add_node(nNodes);

        // costs
        MultiArray<2, FLOAT > costNodesS(Shape2D(nNodes, 1), 0.0);
        MultiArray<2, FLOAT > costNodesT(Shape2D(nNodes, 1), 0.0);

        // compute cost: flux maximization
        if (typeTLink & TLINK_METHOD_FLUX != 0) {
            std::vector<Shape > neighbors = createNeighborhood<DIM >(neighborhood, true);
            MultiArrayView<2, FLOAT > vfV(
                    Shape2D(data.elementCount(), gvf.elementCount() / data.elementCount()),
                    gvf.data());

            if (verbose)
                std::cerr << prefix << "compute flux cost, lambda = " << lambdaFlux << std::endl;

            FLOAT costTMin = 1e10;
            FLOAT costTMax = -1e10;
            
            int indEdge = 0;
            for (int indP = 0; indP < nNodes; indP ++) {
                Shape shapeP = data.scanOrderIndexToCoordinate(indP);

                // wall through all neighbors
                FLOAT costT = 0;
                for (int indN = 0; indN < neighbors.size(); indN ++) {
                    Shape shapeQ = shapeP + neighbors[indN];
                    if (!data.isInside(shapeQ)) 
                        continue ;
                    int indQ = data.coordinateToScanOrderIndex(shapeQ);

                    // compute n-link cost
                    costT += costTLinkFlux<DIM, FLOAT, FLOAT >(shapeP, shapeQ, vfV.bindInner(indQ));
                }
//                costT /= neighborhood;
                if (costT < 0)
                    costNodesS[indP] += lambdaFlux * (- costT);
                else
                    costNodesT[indP] += lambdaFlux * (costT);
                
                // save the min/max costN
                costTMin = std::min(costTMin, costT);
                costTMax = std::max(costTMax, costT);
            }
            if (verbose)
                std::cerr << prefix << "\tmax = " << costTMax << ", min = " << costTMin << std::endl;
        }

        // compute cost: probability map
        if (typeTLink & TLINK_METHOD_PROBMAP != 0) {
            if (verbose)
                std::cerr << prefix << "compute probmap cost, lambda = " << lambdaProbMap << std::endl;
            
            FLOAT costTMin = 1e10;
            FLOAT costTMax = -1e10;
            int indEdge = 0;
            for (int indP = 0; indP < nNodes; indP ++) {
                Shape shapeP = data.scanOrderIndexToCoordinate(indP);
                FLOAT costT = probmap[indP];
                costNodesS[indP] += lambdaProbMap * (1 - costT);
                costNodesT[indP] += lambdaProbMap * costT;
                
                // save the min/max costN
                costTMin = std::min(costTMin, costT);
                costTMax = std::max(costTMax, costT);
            }
            
            if (verbose)
                std::cerr << prefix << "\tmax = " << costTMax << ", min = " << costTMin << std::endl;
        }

        // compute cost: boundary cue
        if (typeTLink & TLINK_METHOD_BOUNDARYCUE != 0) {
            if (verbose)
                std::cerr << prefix << "compute boundary cue cost, lambda = " << lambdaBoundaryCue << std::endl;

            FLOAT costTMin = 1e10;
            FLOAT costTMax = -1e10;
            
            int indEdge = 0;
            for (int indP = 0; indP < nNodes; indP ++) {
                Shape shapeP = data.scanOrderIndexToCoordinate(indP);
                FLOAT costT = bdcue[indP];
                costNodesS[indP] += lambdaBoundaryCue * costT;
                
                // save the min/max costN
                costTMin = std::min(costTMin, costT);
                costTMax = std::max(costTMax, costT);
            }
            
            if (verbose)
                std::cerr << prefix << "\tmax = " << costTMax << ", min = " << costTMin << std::endl;
        }

        // compute cost: seeded
        if (typeTLink & TLINK_METHOD_SEEDED != 0) {
            if (verbose)
                std::cerr << prefix << "compute seed cost" << std::endl;

            int indEdge = 0;
            for (int indP = 0; indP < nNodes; indP ++) {
                Shape shapeP = data.scanOrderIndexToCoordinate(indP);
                if (mask[indP] == 0)
                    continue;

                if (mask[indP] == 1) {                      // 1 - fixed to the foreground
                    costNodesS[indP] = 0;
                    costNodesT[indP] = 1e20;
                }
                else if (mask[indP] == 2) {                  // 2 - fixed to the background
                    costNodesS[indP] = 1e20;
                    costNodesT[indP] = 0;
                }
            }
        }

        // set the weights in the graph
        for (int indP = 0; indP < nNodes; indP ++) 
            g->add_tweights(indP, costNodesS[indP], costNodesT[indP]);
    }

    // add n-link costs
    template <int DIM, class INT, class FLOAT > void initializeNLinkCosts(
            Graph<FLOAT, FLOAT, FLOAT > *g, 
            MultiArray<DIM, FLOAT > &data,
            MultiArray<DIM, INT > &mask,
            MultiArray<DIM+1, FLOAT > &gvf)
    {
        if (verbose)
            std::cerr << prefix << "initialize n-link costs ..." << std::endl;

        typedef typename MultiArrayView<DIM, double >::difference_type Shape;
        typedef MultiArrayView<2, double >::difference_type Shape2D;

        std::vector<Shape > neighbors = createNeighborhood<DIM >(neighborhood);

        int nNodes = data.elementCount();    
        int nEdges = estimateNumberOfEdges<DIM >(data.shape(), neighborhood);

        // parameters
        MultiArrayView<2, FLOAT > vfV;
        if (gvf.elementCount() > 0) {
            vfV = MultiArrayView<2, FLOAT >(
                Shape2D(data.elementCount(), gvf.elementCount() / data.elementCount()),
                gvf.data());
        }

        // compute cost: gaussia
        for (int indP = 0; indP < nNodes; indP ++) {
            Shape shapeP = data.scanOrderIndexToCoordinate(indP);
            
//            if (mask[indP] !=0)
//                continue;

            // wall through all neighbors
            for (int idxN = 0; idxN < neighbors.size(); idxN ++) {
                Shape shapeQ = shapeP + neighbors[idxN];
                if (!data.isInside(shapeQ))
                    continue ;

                int indQ = data.coordinateToScanOrderIndex(shapeQ);

                FLOAT costPtoQ = 0;	
                FLOAT costQtoP = 0;
                if (mask[indP] !=0 && mask[indQ] != 0) 
                    continue;

                // compute the gaussian
                if (typeNLink & NLINK_METHOD_GAUSSIAN != 0) {       // cost_n = exp(-(x1-x2)^2/sigma^2)
                    FLOAT cost = lambdaGaussian * costNLinkGaussian<FLOAT, FLOAT > (data[indP], data[indQ], sigmaNoise);
                    costPtoQ += cost;
                    costQtoP += cost;
                }

                // compute the shape prior
                if (typeNLink & NLINK_METHOD_SHAPE != 0) {         // cost_n = 1 / <n_pq, v_p>
                    costPtoQ += lambdaShape * 
                            costNLinkShape<DIM, FLOAT, FLOAT >(shapeP, shapeQ, vfV.bindInner(indP), 0);
                    costQtoP += lambdaShape * 
                            costNLinkShape<DIM, FLOAT, FLOAT >(shapeQ, shapeP, vfV.bindInner(indP), 0);
                }

                // add this edge to the graph
                g->add_edge(indP, indQ, costPtoQ, costQtoP);
            }
        }
    }

    template <int DIM, class FLOAT, class INT > void run(
            MultiArray<DIM, FLOAT > &data, 
            MultiArray<DIM, INT > &mask, 
            MultiArray<DIM, FLOAT > &probmap, 
            MultiArray<DIM+1, FLOAT > &gvf, 
            MultiArray<DIM, FLOAT > &bdcue,
            MultiArray<DIM, INT > &seg)
    {
        typedef typename MultiArray<DIM, FLOAT >::difference_type Shape;
        typedef typename MultiArray<DIM+1, FLOAT >::difference_type ShapeGVF;

        if (seg.elementCount() == 0)
            seg.reshape(data.shape(), 0);
        
        // pay more attention to the memory consumption here: graph cut is memory-wise expensive        
        Graph<FLOAT, FLOAT, FLOAT >* g = createGraph<3, FLOAT >(data);

        initializeTLinkCosts<3, INT, FLOAT >(g, data, mask, probmap, gvf, bdcue); 
        probmap.reshape(Shape(1, 1, 1), 0);
        bdcue.reshape(Shape(1, 1, 1), 0);

        initializeNLinkCosts<3, INT, FLOAT >(g, data, mask, gvf);
        gvf.reshape(ShapeGVF(1, 1, 1, 1), 0);
        mask.reshape(Shape(1, 1, 1), 0);
        data.reshape(Shape(1, 1, 1), 0);

        callMaxFlow<3, FLOAT, INT >(g, seg);

        // free memory
        delete g;
    }

private:
    int typeTLink, typeNLink, neighborhood;
    int verbose;
    std::string prefix;
    float lambdaFlux, lambdaProbMap, lambdaBoundaryCue, lambdaGaussian, lambdaShape, sigmaNoise;
};

#endif /* __CT_SEGMENTATION_GC__ */
