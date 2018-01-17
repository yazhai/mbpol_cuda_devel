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

#ifdef _USE_GSL
#include <gsl/gsl_cblas.h>
#elif _USE_MKL
#include <mkl_cblas.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif 

//use these methods if libraries are employed
#if defined (_USE_GSL) || defined (_USE_MKL)
double calcDistance(double* atom1, double* atom2);

float calcDistance(float* atom1, float* atom2);
#endif

//calculate the distance between two xyz coordinates (euclidian norm)
template <typename T>
T calcDistance(T* atom1, T* atom2){
	T diff[3] = {atom1[0]-atom2[0], atom1[1]-atom2[1], atom1[2] - atom2[2]};
	return sqrt(diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]);
}


//load the data from xyz file to data variable 
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
	
	//store all coordinates in mtx vector
     while(getline(ifs, line)){
		
		onelinedata.clear();
		lineVec.clear();
		std::istringstream ss(line);
	
		while(ss>>tmp){
			lineVec.push_back(tmp);
		}

		//xyz data starts with letter of atom, if not xyz skip line
		if(isdigit(lineVec[0][0]) || lineVec[0][0] == '-'){
			continue;
		}
		
		//get the coordinates from string and push them onto mtx
		lineVec.erase(lineVec.begin());
		for(auto it = lineVec.begin(); it!= lineVec.end(); it++){
			onelinedata.push_back(stod(*it));
		}
		mtx.push_back(onelinedata);

	}
	
	//find how many coordinates are read, and copy mtx vector into data 2d array(this is the output)
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
	
	//get xyzData and the number of coordinates from file	
	size_t xyzRows;	
	loadXYZ(xyzData, xyzRows, xyzFile);
	
	//rows = # of trimers ( or other groups of atoms)
	//cols = # of distances that must be calculated between all atoms that belong to the group
	//distRows = xyzRows/numAtoms;
	//distCols = (size_t)(numAtoms*(numAtoms-1)/2);
	//init_mtx_in_mem<T>(distData,distRows,distCols);

	distCols = xyzRows/numAtoms;
	distRows = (size_t)(numAtoms*(numAtoms-1)/2);
	init_mtx_in_mem<T>(distData,distRows,distCols);

	//Calculate Distances from XYZ coordinates
	int currentIndex = 0;
	for(int ii=0;ii<numAtoms;ii++){
		for(int jj=ii+1;jj<numAtoms;jj++){
			#ifdef _OPENMP
			#pragma omp parallel for simd shared(distCols,distData,currentIndex,xyzData,numAtoms,ii,jj)
			#endif
			for(int N=0;N<distCols;N++){
				distData[currentIndex][N] = calcDistance(xyzData[N*numAtoms+ii], xyzData[N*numAtoms+jj]);
			}
			currentIndex++;
		}
	}
}

		
	

#endif 


