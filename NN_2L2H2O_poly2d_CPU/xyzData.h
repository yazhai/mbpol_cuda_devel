#ifndef XYZDATA_H
#define XYZDATA_H

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cmath>
#include "utility.h"
// Define the cblas library 

#include <gsl/gsl_cblas.h>

template <typename T>
T calcDistance(T* atom1, T*atom2){
	T diff[6] = {atom1[0]-atom2[0], atom1[1]-atom2[1], atom1[2] - atom2[2]};
	return cblas_dnrm2(3,diff,1);
}

template <typename T>
int loadXYZ(T ** & data, size_t & rows, const char* xyzFile){
	std::ifstream ifs(xyzFile);
     std::string line;
	std::vector<std::string> lineVec;

     matrix_by_vector_t<T> mtx;
	std::vector<T> onelinedata;
	std::string tmp;
	size_t cols = 3;	

	int i=0;
     while(getline(ifs, line)){

		onelinedata.clear();
		lineVec.clear();
		std::istringstream ss(line);

		while(ss>>tmp){
			lineVec.push_back(tmp);
		}

		//xyz data starts with letter of atom
		if(isdigit(lineVec[0][0]) || lineVec[0][0] == '-'){
			continue;
		}
		
		lineVec.erase(lineVec.begin());
		for(auto it = lineVec.begin(); it!= lineVec.end(); it++){
			onelinedata.push_back(stod(*it));
		}
		mtx.push_back(onelinedata);

	}
	rows=mtx.size();
     if (rows > 0){  
       	init_mtx_in_mem<T>(data, rows, cols);  
             
         	for(int ii=0; ii<rows; ii++){
            	copy(mtx[ii].begin(), mtx[ii].end(), data[ii]);       
        	}          
 	} else {
     	std::cout << " No Data is read from file as 2D array" << std::endl;
		return -1;
  	}
   	mtx.clear();
	return 0; 	
}

template <typename T>
int getDist(T ** & xyzData, const char* xyzFile, T ** & distData, size_t & distRows, size_t & distCols, size_t numAtoms){
	size_t xyzRows;	
	loadXYZ(xyzData, xyzRows, xyzFile);
	distRows = xyzRows/numAtoms;
	distCols = (size_t)(numAtoms*(numAtoms-1)/2);
	init_mtx_in_mem<T>(distData,distRows,distCols);
	int currentIndex;
	for(int N=0;N<distRows;N++){
		currentIndex=0;
		for(int ii=0;ii<numAtoms;ii++){
			for(int jj=ii+1;jj<numAtoms;jj++){
				distData[N][currentIndex] = calcDistance(xyzData[N*numAtoms+ii], xyzData[N*numAtoms+jj]);
				currentIndex++;
			}
		}
	}
}

		
	

#endif 


