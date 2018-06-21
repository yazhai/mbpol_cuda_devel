#ifndef GFUNCTION_H
#define GFUNCTION_H

#include <cstdlib>
#include <vector>
#include <string>
#include <map>
#include <limits>
#include <math.h>
#include <algorithm>

#include "utility.h"
#include "atomTypeID_v2.h"
#include "readGparams_v2.h"

#include "timestamps.h"

// Define the cblas library 
#ifdef _USE_GSL
#include <gsl/gsl_cblas.h>
#elif _USE_MKL
#include <mkl_cblas.h>
#else 
//#include <gsl/gsl_cblas.h>
#endif


#ifdef _OPENMP
#include <omp.h>
#endif 


namespace MBbpnnPlugin{



const int COL_RAD_RS = 3;
const int COL_RAD_ETA = 2; 
const int COL_RAD_TOTAL = 5;

const int COL_ANG_ETA = 2;
const int COL_ANG_ZETA = 4;
const int COL_ANG_LAMBD=3;
const int COL_ANG_TOTAL=6;



template <typename T>
class Gfunction_t : public MBbpnnPlugin::atom_Type_ID_t<T> {
private:

//===========================================================================================
//
// Elementwise functions
T cutoff(T R, T R_cut=10.0) ; 

T get_cos(T Rij, T Rik, T Rjk) ;

T get_Gradial(T  Rij, T Rs, T eta) ;

T get_Gangular(T Rij, T Rik, T Rjk, T eta, T zeta, T lambd) ;

T get_dist(size_t atom1, size_t atom2, size_t dimer_idx) ;


//===========================================================================================
// Vectorized functions
//
// These functions are not vectorized at the moment, 
// but API are left as vectorized form for consecutive memory utilization and 
// future compatible possibility with other linear algebra libraries.
void cutoff(T* rst, T* Rij, size_t n, T R_cut=10) ;

void get_cos(T * rst, T * Rij, T * Rik, T * Rjk, size_t n) ;

void get_Gradial(T* rst, T* Rij, size_t n, T Rs, T eta, T R_cut=10 ) ;

void get_Gradial_add(T* rst, T* Rij, size_t n, T Rs, T eta , T* tmp = nullptr ) ;
 
void get_Gangular(T* rst, T* Rij, T* Rik, T* Rjk, size_t n, T eta, T zeta, T lambd ) ;

void get_Gangular_add(T* rst, T* Rij, T* Rik, T* Rjk, size_t n, T eta, T zeta, T lambd, T* tmp = nullptr ) ;


// Helper functions for gradient: 
//
// e.g. get_dGdD = local gradient dG/dD
//
T get_dGdR_rad(T Rij, T Rref, T eta) ;

T get_dGdR_ang_IJ(T Rij, T Rik, T Rjk, T eta, T zeta, T lambd) ;

T get_dGdR_ang_JK(T Rij, T Rik, T Rjk, T eta, T zeta, T lambd) ;


T get_dRdX(T Rij, T Xi, T Xj) ;

void get_dRdX_byID(T *  dRdX ,  idx_t atom1, idx_t atom2, size_t dimer_idx);

// helper functions in creating G-fn
void init_G();
void scale_G(std::string tag);
void scale_G_grd(std::vector<T**> dfdG);

// Timers for benchmarking
timers_t timers;
timerid_t id, id1, id2 , id3;

std::map<std::string, T*> G_SCALE_CONST; // reserved scaling factor for each atom after NN
std::map<std::string, T*> G_SCALE_VAR; // read-in scaling factor for each atom after NN
void load_resv_scales(std::string tag); // saving the scaling factors to the reserved vector;


// the following functions offers a switch_factor that is applied to the energy from NN (works as an energy scale factor)
T base_fswitch(T ri, T rf, T r);
T base_dfdswitch(T ri, T rf, T r);  
void fswitch_2b(T ri, T rf, idx_t at1, idx_t at2);
void fswitch_3b(T ri, T rf, idx_t at1, idx_t at2, idx_t at3);

public:


T** xyz;                    //xyz data of atoms used for calculation; duplicated from XYZ in atom_Type_ID class ;
T** dfdxyz;                 //function gradient on xyz of atoms, the same size of xyz
T*  switch_factor;            // switch function factor applies to energy from NN
T*  dfdswitch;                // gradient of switch function

Gparams_t<T> gparams;                 // G-fn paramter class

std::vector<T**> G;       // G-fn matrix
std::vector<T*>  G_scale; // Scaling factor for each atom after NN
std::vector<size_t> G_param_max_size;        //max size of parameters for each atom type


// constructor, destructor
Gfunction_t();
~Gfunction_t();


// file I/O utilities
void load_xyzfile(const char* file);
void load_paramfile(const char* file);
void load_seq(const char* _seqfile);

// default 2h2o
void load_paramfile_2h2o_default();
void load_paramfile_3h2o_default();
void load_seq_2h2o_3h2o_default();
void load_scale_2h2o_default();
void load_scale_3h2o_default();
//=================================================================================
// G-function assembly functions
void make_G();

void make_G_from_files(const char* _xyzFile, const char * _paramfile, const char* _ordfile);  // 

void make_grd(std::vector<T**> dfdG); // get gradient by current model informations


void cal_switch(int flag, idx_t at1 =0, idx_t at2=3, idx_t at3=6);

void get_dfdx_from_switch(int flag, T* e, idx_t at1 =0, idx_t at2=3, idx_t at3=6);

};


extern template class Gfunction_t<float>;
extern template class Gfunction_t<double>;


}; // end of namespace MBbpnnPlugin
#endif
