/*
 *  helpers.cpp
 *  vigra
 *
 *  Created by Luca Fiaschi on 3/11/11.
 *  Copyright 2011 Heidelberg university. All rights reserved.
 *
 */

//This file contains some helper functios for the objectfeatures modules


#include <vigra/multi_array.hxx>


using namespace vigra;


template <class holder>
NumpyAnyArray copyme(const holder& vec){
	
	
	MultiArrayShape<2>::type shape(vec.size(),3);
	
	
	NumpyArray<2, unsigned int> res(shape);
	
	
	for (int i =0;i<shape[0];i++){
		res(i,0)=vec[i][0];
		res(i,1)=vec[i][1];
		res(i,2)=vec[i][2];
		}
		
	
	return res;
}





template < class PixelType >
boost::python::dict
pythonDictObjects3D(NumpyArray<3, Singleband<PixelType> > image){

	
	{ PyAllowThreads _pythread;
		
		
		MultiArrayShape<3>::type shape(image.shape());
		
		
		typedef std::vector<MultiArrayShape<3>::type > holder;
		
		std::map< PixelType , holder > d;
		
		
		
		typename std::map< PixelType , holder >::iterator it;
		
		
		MultiArrayShape<3>::type p(0,0,0);
		
		
		
		
		for (p[2]=0; p[2]<shape[2]; ++p[2]){
			for(p[1]=0; p[1]<shape[1]; ++p[1])
			{
				for(p[0]=0; p[0]<shape[0]; ++p[0])
				{	
					
					d[image[p]].push_back(p);
					
				}
				
				
			}
		}
		
		boost::python::dict pyd;
		
		
		
		
		for(it=d.begin();it!=d.end();it++){
			
			pyd[it->first]=copyme(it->second);
			
			
		}
	
	
	return pyd;
   }
}	




	







