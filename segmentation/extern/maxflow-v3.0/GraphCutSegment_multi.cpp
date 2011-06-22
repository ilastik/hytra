/**************************************************************
GRAPHCUTSEGMENT_multi.CPP - Main graph cuts segmentation C++ function
Using a-expansion

Huan Xu
6,March,2009
***************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "mex.h"
#include "graph.h"
#include "helper_functions.h"
#include "vigra/random_forest.hxx"
#include "vigra/matlab.hxx"
#include "vigra/random_forest_impex.hxx"

int MakeAdjacencyList(double *L, int N, int numCols, int numRows,vector<double> initfVec, vector< set<int> > &neighbors, vector < vector<int> > &PositionPQ)
{
	int i,j;
	
	neighbors.clear();

	// Declaring the Adjacency list
	for(i=0;i<N;i++)
	{
		set<int> tmp;
		tmp.clear();
		neighbors.push_back(tmp);
	}
	int apq = N;
	
	//printf("Adjacency code starts... \n");

	// Filling up the adjacency list
	for(i=0;i<numRows;i++)
		for(j=0;j<numCols;j++)
		{
			int thisLabel = (int) L[j*numRows + i];	// Label of the current pixel
		
			if(thisLabel==-1)							// If this is A watershed Label --> A boundary Pixel
			{

				// 4-neighborhood
				int UpLabel    = (i>0)            ? (int) L[j*numRows + (i-1)] : thisLabel;
				int DownLabel  = (i<(numRows-1))  ? (int) L[j*numRows + (i+1)] : thisLabel;
				int LeftLabel  = (j>0)            ? (int) L[(j-1)*numRows + i] : thisLabel;
				int RightLabel = (j<(numCols-1))  ? (int) L[(j+1)*numRows + i] : thisLabel;

				// 8-neighborhood
				int NWLabel    = (i>0 && j>0)						? (int) L[(j-1)*numRows + (i-1)] : thisLabel;
				int NELabel    = (i>0 && j<(numCols-1))				? (int) L[(j+1)*numRows + (i-1)] : thisLabel;
				int SELabel    = (i<(numRows-1) && j<(numCols-1))	? (int) L[(j+1)*numRows + (i+1)] : thisLabel;
				int SWLabel    = (i<(numRows-1) && j>0)				? (int) L[(j-1)*numRows + (i+1)] : thisLabel;

                                // 256-neighborhood
                int D1 = (i<(numRows-1)&&j>1)?(int)L[(j-2)*numRows+(i+1)]:thisLabel;
                int D2 = (i<(numRows-1)&&j<(numCols-2))?(int)L[(j+2)*numRows+(i+1)]:thisLabel;
                int D3 = (i>0&&j>1)?(int)L[(j-2)*numRows+(i-1)]:thisLabel;
                int D4 = (i>0&&j<(numCols-2))?(int)L[(j+2)*numRows+(i-1)]:thisLabel;
                int D5 = (i<(numRows-2)&&j>0)?(int)L[(j-1)*numRows+(i+2)]:thisLabel;
                int D6 = (i>1&&j>0)?(int)L[(j-1)*numRows+(i-2)]:thisLabel;
                int D7 = (i<(numRows-2)&&j<(numCols-1))?(int)L[(j+1)*numRows+(i+2)]:thisLabel;
                int D8 = (i>1&&j<(numCols-1))?(int)L[(j+1)*numRows+(i-2)]:thisLabel;


				set<int> surround;						// Surrounding Labels of this boundary Label
				
				surround.insert(UpLabel);
				surround.insert(DownLabel);
				surround.insert(LeftLabel);
				surround.insert(RightLabel);
				surround.insert(NWLabel);
				surround.insert(NELabel);
				surround.insert(SELabel);
				surround.insert(SWLabel);
                surround.insert (D1);
 				surround.insert (D2);
 				surround.insert (D3);
 				surround.insert (D4);
 				surround.insert (D5);
 				surround.insert (D6);
 				surround.insert (D7);
 				surround.insert (D8);

				surround.erase(-1);						// Removing other boundary pixels
				set<int>::iterator pIter, qIter;    //adding each other , for example, if the surround is (2,4,6,7), then the neighbors[2]--4,6,7. neighbors[4]--2,6,7..neighbors[6]--2,4,7.. neighbors[7]--2,4,6
				for (pIter = surround.begin(); pIter != surround.end(); pIter++)
					for (qIter = pIter; qIter != surround.end(); qIter++)
					{
						if (initfVec[*pIter] == initfVec[*qIter] )  
						   { neighbors[*pIter].insert(*qIter); neighbors[*qIter].insert(*pIter); }
						else
						{	set<int> tmp;		tmp.clear();		neighbors.push_back(tmp);
							neighbors[*pIter].insert(apq);  neighbors[*qIter].insert(apq);
							neighbors[apq].insert(*pIter); neighbors[apq].insert(*qIter);
                            
                            
                            /////////////recorde the position of the *piter, *qIter------vector< vector<int> > PositionPQ
                            vector<int> tmp2; tmp2.clear();  
                               
                            if (*pIter==UpLabel){   tmp2.push_back (i-1);  tmp2.push_back (j); }
                            else if(*pIter==DownLabel){ tmp2.push_back (i+1);  tmp2.push_back (j); }
                            else if(*pIter==LeftLabel){ tmp2.push_back (i);  tmp2.push_back (j-1); }
                            else if(*pIter==RightLabel){ tmp2.push_back (i);  tmp2.push_back (j+1);  }                            
                            else if(*pIter==NWLabel){  tmp2.push_back (i-1);  tmp2.push_back (j-1); }
                            else if(*pIter==NELabel){ tmp2.push_back (i-1);  tmp2.push_back (j+1);  }
                            else if(*pIter==SELabel){ tmp2.push_back (i+1);  tmp2.push_back (j+1); }
                            else if(*pIter==SWLabel){tmp2.push_back (i+1);  tmp2.push_back (j-1);  }
                            else if(*pIter==D1){ tmp2.push_back (i+1);  tmp2.push_back (j-2); }
                            else if(*pIter==D2){ tmp2.push_back (i+1);  tmp2.push_back (j+2); }
                            else if(*pIter==D3){ tmp2.push_back (i-1);  tmp2.push_back (j-2); }
                            else if(*pIter==D4){ tmp2.push_back (i-1);  tmp2.push_back (j+2); }
                            else if(*pIter==D5){ tmp2.push_back (i+2);  tmp2.push_back (j-1); }
                            else if(*pIter==D6){ tmp2.push_back (i-2);  tmp2.push_back (j-1); }
                            else if(*pIter==D7){ tmp2.push_back (i+2);  tmp2.push_back (j+1); }
                            else if(*pIter==D8){ tmp2.push_back (i-2);  tmp2.push_back (j+1); }
                            
                            if (*qIter==UpLabel){   tmp2.push_back (i-1);  tmp2.push_back (j); }
                            else if(*qIter==DownLabel){ tmp2.push_back (i+1);  tmp2.push_back (j); }
                            else if(*qIter==LeftLabel){ tmp2.push_back (i);  tmp2.push_back (j-1); }
                            else if(*qIter==RightLabel){ tmp2.push_back (i);  tmp2.push_back (j+1);  }                            
                            else if(*qIter==NWLabel){  tmp2.push_back (i-1);  tmp2.push_back (j-1); }
                            else if(*qIter==NELabel){ tmp2.push_back (i-1);  tmp2.push_back (j+1);  }
                            else if(*qIter==SELabel){ tmp2.push_back (i+1);  tmp2.push_back (j+1); }
                            else if(*qIter==SWLabel){tmp2.push_back (i+1);  tmp2.push_back (j-1);  }
                            else if(*qIter==D1){ tmp2.push_back (i+1);  tmp2.push_back (j-2); }
                            else if(*qIter==D2){ tmp2.push_back (i+1);  tmp2.push_back (j+2); }
                            else if(*qIter==D3){ tmp2.push_back (i-1);  tmp2.push_back (j-2); }
                            else if(*qIter==D4){ tmp2.push_back (i-1);  tmp2.push_back (j+2); }
                            else if(*qIter==D5){ tmp2.push_back (i+2);  tmp2.push_back (j-1); }
                            else if(*qIter==D6){ tmp2.push_back (i-2);  tmp2.push_back (j-1); }
                            else if(*qIter==D7){ tmp2.push_back (i+2);  tmp2.push_back (j+1); }
                            else if(*qIter==D8){ tmp2.push_back (i-2);  tmp2.push_back (j+1); }
                                                    
                                                
                            PositionPQ.push_back(tmp2);
                            
                            
                            /////////////////////
                            apq = apq+1;
                            
                            
                            
                            
                            
						}
					}

			}
		}

    for(i=0;i<apq;i++)
	{
		(neighbors[i]).erase(i);				// Removing Self Edges from the adjacency list
		
	}

	return apq;
}

Graph* MakeGraph( vector< set<int> > neighbors, vector < vector <int> > PositionPQ, vector< vector<double> > MeanColors, vector< vector<double> >  Pixel_ProbVec ,vector<double> initfVec, int N, int totalnum, double alpha,double lambda, Graph::node_id *nodes,vector<double> Pixel_alpha_ProbVec,double color,std::auto_ptr<vigra::RandomForest<double> > Pixels_Pairwise_RF )
{
	Graph *G = new Graph();
	int i;
	set<int>::iterator pIter;
	const int K = 10000;					// Edge weight for infinity -- All other edge weights less than 1 -- so it suffices to have this weight as more than sum of all other edges, i.e. >8
	
	vector<double> ForeEdges, BackEdges;
	ForeEdges.clear();
	BackEdges.clear();
	

	/****** Making Terminal Edge Weights ******/
	for(i=0;i<N;i++)
	{
		if (initfVec[i]==alpha)
		{
			ForeEdges.push_back(Pixel_ProbVec[i][int(alpha)-1]);
			BackEdges.push_back(K);   

		}
		else
		{
			ForeEdges.push_back(Pixel_ProbVec[i][int(alpha)-1]);
			BackEdges.push_back(Pixel_ProbVec[i][int(initfVec[i])-1]);

		}
	}

	for(i=0;i<totalnum;i++)     // now, in fact, G->nodeblcok has the same space with the nodes
		nodes[i] = G -> add_node();

	// Setting Terminal Edge Weights
	for(i=0;i<N;i++)
		G -> set_tweights(nodes[i], ForeEdges[i], BackEdges[i]);


	// Setting Neighboring Edge Weights
	for(i=0;i<N;i++)
		for (pIter = neighbors[i].begin(); pIter != neighbors[i].end(); pIter++)
		{
			int tmpN = *pIter;
			int t= (int(initfVec[i]-alpha))==0?0:1;

			double edge = t/(0.0001+Pixel_alpha_ProbVec [i]) ;
		
			G -> add_edge(nodes[i], nodes[tmpN], lambda*edge, lambda*edge);	
		}

	//mexPrintf ("not really\n");

     for (i=N; i< totalnum; i++)
	 {
		 pIter = neighbors[i].begin();
		 int tmp = *pIter;
         pIter++;
		 int tmq = *pIter;

	     double e[2]={0,0};
		/* double FMeanColors[7];
         int j;
		 for (j=0;j<3;j++)
			 FMeanColors[j]= MeanColors [tmp][j];
		 for (j=3;j<6;j++)
             FMeanColors[j]= MeanColors [tmq][j-3];
		 FMeanColors[6]=1;
         vigra::MultiArrayView<2,double> test(vigra::MultiArrayShape<2>::type(1, 2), e);
         Pixels_Pairwise_RF->predictProbabilities ( vigra::MultiArrayView<2,double>(vigra::MultiArrayShape<2>::type(1, 7), FMeanColors), test);
*/
         double FMeanColors[10]={0};
         int j;
		 for (j=0;j<3;j++)
			 FMeanColors[j]= MeanColors [tmp][j];
         
         FMeanColors[3] = double(PositionPQ [i-N][0]);
         FMeanColors[4] = PositionPQ [i-N][1];
         
		 for (j=5;j<8;j++)
             FMeanColors[j]= MeanColors [tmq][j-5];
         
         FMeanColors[8] = PositionPQ [i-N][2];
         FMeanColors[9] = PositionPQ [i-N][3];
		// FMeanColors[6]=1;
         vigra::MultiArrayView<2,double> test(vigra::MultiArrayShape<2>::type(1, 2), e);
         Pixels_Pairwise_RF->predictProbabilities ( vigra::MultiArrayView<2,double>(vigra::MultiArrayShape<2>::type(1, 10), FMeanColors), test);
         
         
		 G -> set_tweights(nodes[i],0, lambda*e[1]);   // no edge is 0 ??? should be 0, if it is 0, it couldnot seg this backedge
	 }

	//printf("Graph Made... \n");
	return G;
}

double SegmentImage(double *L, Graph *G, int numRows, int numCols, double *SegImage, Graph::node_id *nodes, double *newf,vector<double> initfVec,int N,double alpha,double color)
{
	int i,j,k;
	Graph::flowtype flow = G -> maxflow();

		for (i=0; i< N; i++)
	{
		if(G->what_segment(nodes[i]) == Graph::SOURCE)
			newf[i] = alpha;
		else
			newf[i] = initfVec[i];
	}


	for(i=0;i<numRows;i++)
		for(j=0;j<numCols;j++)
		{
			int thisLabel = (int) L[j*numRows + i];
			if(thisLabel>=0)								// If not Boundary pixel
			{
				if (G->what_segment(nodes[thisLabel]) == Graph::SOURCE)		// Do the classification...
                	SegImage[j*numRows + i] = alpha;
				else
					SegImage[j*numRows + i] = initfVec[thisLabel];
			}
			else
				SegImage[j*numRows + i] = 0.0;								// Label the boundary pixels as background for now...
		}

	// Correctly label the boundary pixels ... Label as foreground if any of the neighbors is foreground
	for(i=0;i<numRows;i++)													
		for(j=0;j<numCols;j++)
		{
			int thisLabel = (int) L[j*numRows + i];
			if (thisLabel==-1)
			{
				int UpLabel    = (i>0)            ? (int) L[j*numRows + (i-1)] : thisLabel;
				int DownLabel  = (i<(numRows-1))  ? (int) L[j*numRows + (i+1)] : thisLabel;
				int LeftLabel  = (j>0)            ? (int) L[(j-1)*numRows + i] : thisLabel;
				int RightLabel = (j<(numCols-1))  ? (int) L[(j+1)*numRows + i] : thisLabel;

				// 8-neighborhood
				int NWLabel    = (i>0 && j>0)						? (int) L[(j-1)*numRows + (i-1)] : thisLabel;
				int NELabel    = (i>0 && j<(numCols-1))				? (int) L[(j+1)*numRows + (i-1)] : thisLabel;
				int SELabel    = (i<(numRows-1) && j<(numCols-1))	? (int) L[(j+1)*numRows + (i+1)] : thisLabel;
				int SWLabel    = (i<(numRows-1) && j>0)				? (int) L[(j-1)*numRows + (i+1)] : thisLabel;

                                // 256-neighborhood
                                int D1 = (i<(numRows-1)&&j>1)?(int)L[(j-2)*numRows+(i+1)]:thisLabel;
                                int D2 = (i<(numRows-1)&&j<(numCols-2))?(int)L[(j+2)*numRows+(i+1)]:thisLabel;
                                int D3 = (i>0&&j>1)?(int)L[(j-2)*numRows+(i-1)]:thisLabel;
                                int D4 = (i>0&&j<(numCols-2))?(int)L[(j+2)*numRows+(i-1)]:thisLabel;
                                int D5 = (i<(numRows-2)&&j>0)?(int)L[(j-1)*numRows+(i+2)]:thisLabel;
                                int D6 = (i>1&&j>0)?(int)L[(j-1)*numRows+(i-2)]:thisLabel;
                                int D7 = (i<(numRows-2)&&j<(numCols-1))?(int)L[(j+1)*numRows+(i+2)]:thisLabel;
                                int D8 = (i>1&&j<(numCols-1))?(int)L[(j+1)*numRows+(i-2)]:thisLabel;


				set<int> surround;						// Surrounding Labels of this boundary Label
				
				surround.insert(UpLabel);
				surround.insert(DownLabel);
				surround.insert(LeftLabel);
				surround.insert(RightLabel);
				surround.insert(NWLabel);
				surround.insert(NELabel);
				surround.insert(SELabel);
				surround.insert(SWLabel);
                                surround.insert (D1);
                                surround.insert (D2);
                                surround.insert (D3);
                                surround.insert (D4);
                                surround.insert (D5);
                                surround.insert (D6);
                                surround.insert (D7);
                                surround.insert (D8);




				surround.erase(-1);						// Removing other boundary Labels
				set<int>::iterator pIter;
                
				
				vector<double> tmpColor;
                tmpColor.clear();
				for (k=0;k<=color;k++)
					tmpColor.push_back(0);

                for (pIter = surround.begin(); pIter != surround.end(); pIter++)
                {
                   if(G->what_segment(nodes[*pIter]) == Graph::SOURCE)
	                    tmpColor[int(alpha)]=tmpColor[int(alpha)]+1;
					else
						tmpColor [int(initfVec[*pIter])]= tmpColor [int(initfVec[*pIter])] +1;
                }
                
				double max = 0;
				double mcolor=0;
				for (k=1;k<=color;k++)
					if (tmpColor[k]>max) { max =tmpColor[k]; mcolor = k; } 


                SegImage[j*numRows + i] = mcolor;
			}
		}
		return flow; 
}



void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
	double *L, *MeanColors, *Pixel_Prob, *initf, *palpha, *plambda, *pcolor, *Pixel_alpha_Prob; // Input Arguments

	if(nrhs!=10) {
		mexPrintf("%d \n",nrhs);
		mexErrMsgTxt("Incorrect No. of inputs\n");
	} else if(nlhs!=3) {
		mexErrMsgTxt("Incorrect No. of outputs");
	}
	  
	double *size = mxGetPr (prhs[9]);
	std::auto_ptr<vigra::RandomForest<double> > Pixels_Pairwise_RF = vigra::matlab::importRandomForest<double >(vigra::matlab::ConstCellArray(prhs[8],*size)) ;
	
	double *Energy,*SegImage,*newf;	// Output Argument
	int numRows, numCols, N;	
	int i,j;
	  

	L          = mxGetPr(prhs[0]);
	MeanColors = mxGetPr(prhs[1]);
	Pixel_Prob    = mxGetPr(prhs[2]);
	initf    = mxGetPr(prhs[3]);	
	palpha = mxGetPr(prhs[4]);
	plambda = mxGetPr(prhs[5]);
	pcolor = mxGetPr (prhs[6]);
	Pixel_alpha_Prob = mxGetPr (prhs[7]);


	numCols = mxGetN(prhs[0]);		// Image Size
	numRows = mxGetM(prhs[0]);
	N = mxGetM(prhs[2]);	// Number of pixels (for the region in the watershed.. N!=numCols*numRows)

	double alpha = (*palpha);
	double lambda = (*plambda);
	double color = (*pcolor);

	
	plhs[0] = mxCreateDoubleMatrix(1,1, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(numRows,numCols, mxREAL);	// Memory Allocated for output array
	plhs[2] = mxCreateDoubleMatrix(N,1, mxREAL);		// Memory Allocated for output newf    

	Energy = mxGetPr (plhs[0]);
	SegImage = mxGetPr(plhs[1]);								// variable assigned to the output memory location
	newf = mxGetPr(plhs[2]);
	
	vector< vector<double> > MeanColorsVec;
	MeanColorsVec.clear();
    for(i=0;i<N;i++)
	{
		vector<double> tmp;
		tmp.clear();
		tmp.push_back(MeanColors[int(0*N + i)]);
		tmp.push_back(MeanColors[int(1*N + i)]);
		tmp.push_back(MeanColors[int(2*N + i)]);

		MeanColorsVec.push_back(tmp);
	}


	vector< vector<double> >  Pixel_ProbVec;
    Pixel_ProbVec.clear ();

	for (i=0; i<N;i++)
	{
		vector<double> tmp;
		tmp.clear();

		for (j=0;j<color;j++)
		{
            tmp.push_back (Pixel_Prob[j*N + i]);
		}
		Pixel_ProbVec.push_back(tmp);
	}

	vector<double>   Pixel_alpha_ProbVec; 
    Pixel_alpha_ProbVec.clear ();

	for (i=0; i<N;i++)
	{
		Pixel_alpha_ProbVec.push_back(Pixel_alpha_Prob[i]);
	}

	vector<double> initfVec;
    for(i=0;i<N;i++)
		initfVec.push_back(initf[i]);

	vector< set<int> > neighbors;					// Adjacency list declaration
    vector < vector<int> >PositionPQ;	
	//mexPrintf("Adjacency lists make start \n");
	int totalnum = MakeAdjacencyList(L, N, numCols, numRows, initfVec, neighbors,PositionPQ);	// Adjacency list made
	//mexPrintf("Adjacency lists made end \n");

  /*  for (i=0;i<totalnum-N;i++)
    {
        mexPrintf ("%d %d %d %d\n",PositionPQ[i][0],PositionPQ[i][1],PositionPQ[i][2],PositionPQ[i][3]);
    }
	*/
    
    Graph::node_id *nodes = new Graph::node_id[totalnum];
	
	Graph* G = MakeGraph(neighbors,PositionPQ, MeanColorsVec, Pixel_ProbVec, initfVec, N,totalnum,alpha, lambda, nodes, Pixel_alpha_ProbVec,color, Pixels_Pairwise_RF  );
	//printf("Graph constructed \n");
	
	*Energy = SegmentImage(L, G, numRows, numCols, SegImage, nodes,newf,initfVec,N,alpha,color);
	//printf("Image Segmented \n");
}
