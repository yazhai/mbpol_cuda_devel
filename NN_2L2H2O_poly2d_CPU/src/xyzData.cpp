#include "xyzData.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include "utility.h"

#ifdef _USE_GSL
#include <gsl/gsl_cblas.h>
#elif _USE_MKL
#include <mkl_cblas.h>
#endif

//cblas implementations if library is employed
#if defined (_USE_GSL) || defined (_USE_MKL)
float calcDistance(float* atom1, float* atom2){
	float diff[3] = {atom1[0]-atom2[0], atom1[1]-atom2[1], atom1[2] - atom2[2]};
	return cblas_snrm2(3,diff,1);
}

double calcDistance(double* atom1, double* atom2){
	double diff[3] = {atom1[0]-atom2[0], atom1[1]-atom2[1], atom1[2] - atom2[2]};
	return cblas_dnrm2(3,diff,1);
}
#endif


