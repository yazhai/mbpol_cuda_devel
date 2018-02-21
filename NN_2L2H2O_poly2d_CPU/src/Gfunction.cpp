
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <string.h>
#include <vector>
#include <map>
#include <memory>
#include <cstdlib>
#include <limits>
#include <math.h>
#include <iterator>

#include "Gfunction.h"
#include "readGparams.h"
#include "atomTypeID.h"
#include "utility.h"
#include "timestamps.h"

// Define the cblas library 
#ifdef _USE_GSL
#include <gsl/gsl_cblas.h>
#elif _USE_MKL
#include <mkl_cblas.h>
#else 
//#include <gsl/gsl_cblas.h>
#endif


using namespace std;

//===========================================================================================
// Vectorized functions
//
// These functions are not vectorized at the moment, 
// but API are left as vectorized form for consecutive memory utilization and 
// future compatible possibility with other linear algebra libraries.
// 

// Following functions are defined if cblas library is employed.
#if defined (_USE_GSL) || defined (_USE_MKL)

template <>
void Gfunction_t<double>::get_Gradial_add(double* rst, double* Rij, size_t n, double Rs, double eta, double* tmp ){  
     bool iftmp = false;
     if (tmp == nullptr){
          tmp = new double[n]();
          iftmp = true;             
     }     
     get_Gradial(tmp, Rij, n, Rs, eta);
     cblas_daxpy((const int)n, 1.0, (const double*)tmp, 1, rst, 1);     
     if (iftmp) delete[] tmp;
};

template <>
void Gfunction_t<float>::get_Gradial_add(float* rst, float* Rij, size_t n, float Rs, float eta , float* tmp ){   
     bool iftmp = false;
     if (tmp == nullptr){
          tmp = new float[n]();
          iftmp = true;             
     }        
     get_Gradial(tmp, Rij, n, Rs, eta);
     cblas_saxpy((const int)n, 1.0, (const float*)tmp, 1, rst, 1);   
     if (iftmp) delete[] tmp;  
};


template <>
void Gfunction_t<double>::get_Gangular_add(double* rst, double* Rij, double* Rik, double* Rjk, size_t n, double eta, double zeta, double lambd , double* tmp){
     bool iftmp = false;
     if (tmp == nullptr){
          tmp = new double[n]();
          iftmp = true;             
     }     
     get_Gangular(tmp, Rij, Rik, Rjk, n, eta, zeta, lambd);
     cblas_daxpy((const int)n, 1.0, (const double *)tmp, 1, rst, 1);
     if (iftmp) delete[] tmp;
};


template <>
void Gfunction_t<float>::get_Gangular_add(float* rst, float* Rij, float* Rik, float* Rjk, size_t n, float eta, float zeta, float lambd , float* tmp ){
     bool iftmp = false;
     if (tmp == nullptr){
          tmp = new float[n]();
          iftmp = true;             
     }     
     get_Gangular(tmp, Rij, Rik, Rjk, n, eta, zeta, lambd);
     cblas_saxpy((const int)n, 1.0, (const float *)tmp, 1, rst, 1);
     if (iftmp) delete[] tmp;
};

#endif



/* TEMPLATE METHOD DEFINITIONS */

/*Private Methods*/

//===========================================================================================
//
// Matrix Elementary functions
template <typename T>
T Gfunction_t<T>::cutoff(T R, T R_cut) {
    T f=0.0;
    if (R < R_cut) {    
        //T t =  tanh(1.0 - R/R_cut) ;   // avoid using `tanh`, which costs more than `exp` 
        T t =  1.0 - R/R_cut;        
        t = exp(2*t);
        t = (t-1) / (t+1);                
        f = t * t * t ;        
    }
    return 1 ;
}

template <typename T>
T Gfunction_t<T>::get_cos(T Rij, T Rik, T Rjk) {
    //cosine of the angle between two vectors ij and ik    
    T Rijxik = Rij*Rik ;    
    if ( Rijxik != 0 ) {
          return ( ( Rij*Rij + Rik*Rik - Rjk*Rjk )/ (2.0 * Rijxik) );
    } else {
          return  std::numeric_limits<T>::infinity();
    }
}

template <typename T>
T Gfunction_t<T>::get_Gradial(T  Rij, T Rs, T eta){
     T G_rad = cutoff(Rij);     
     if ( G_rad > 0 ) {
          G_rad *= exp( -eta * ( (Rij-Rs)*(Rij-Rs) )  )  ;
     }
     return G_rad;
}

template <typename T>
T Gfunction_t<T>::get_Gangular(T Rij, T Rik, T Rjk, T eta, T zeta, T lambd){    
    T G_ang = cutoff(Rij)*cutoff(Rik)*cutoff(Rjk);    
    if ( G_ang > 0) {    
          G_ang *=   2 * pow( (1.0 + lambd* get_cos(Rij, Rik, Rjk))/2.0, zeta) 
                     * exp(-eta*  ( (Rij+Rik+Rjk)*(Rij+Rik+Rjk) ) );    
    } 
    return G_ang ;    
}

//Instantiate all non-template class/struct/method of specific type
template double Gfunction_t<double>::cutoff(double R, double R_cut);
template float Gfunction_t<float>::cutoff(float R, float R_cut);

template double Gfunction_t<double>::get_cos(double Rij, double Rik, double Rjk);
template float Gfunction_t<float>::get_cos(float Rij, float Rik, float Rjk);

template double Gfunction_t<double>::get_Gradial(double Rij, double Rs, double eta);
template float Gfunction_t<float>::get_Gradial(float Rij, float Rs, float eta);

template double Gfunction_t<double>::get_Gangular(double Rij, double Rik, double Rjk, double eta, double zeta, double lambd);
template float Gfunction_t<float>::get_Gangular(float Rij, float Rik, float Rjk, float eta, float zeta, float lambd);

