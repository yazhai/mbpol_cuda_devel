
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

#include "Gfunction_cu.cuh"
#include "readGparams_cu.cuh"
#include "atomTypeID.h"
#include "utility.h"
#include "utility_cu.cuh"
#include "timestamps.h"



const char* FLAG_DISTFILE_HEADLINE = "distheadline" ;
const char* FLAG_COLUMN_INDEX_FILE =   "columnfile" ;
const char* FLAG_PARAM_FILE        =    "paramfile" ;
const char* FLAG_ATOM_ORDER_FILE   =      "ordfile" ;
const char* FLAG_CUDA_DEVICE_INDEX =       "device" ;


const int THREDHOLD_COL = -1;
const double THREDHOLD_MAX_VALUE = 60.0;




using namespace std;

//===========================================================================================
// __global__ Vectorized functions
//
template <>
void Gfunction_t<double>::get_Gradial_add(double* rst_d, double* Rij_d, size_t n, double* Rs, double* eta, double* tmp_d ){
     bool iftmp = false;
     if (tmp_d == nullptr){
          init_vec_in_mem_d<double>(tmp_d, n);
          iftmp = true;             
     }           
     get_Gradial(tmp_d, Rij_d, n, Rs, eta);         
     double a = 1.0 ;     
     cublasDaxpy( global_cublasHandle, n, &a, tmp_d, 1, rst_d, 1);
     if (iftmp) clearMemo_d(tmp_d);          
}; 


template <>
void Gfunction_t<float>::get_Gradial_add(float* rst_d, float* Rij_d, size_t n, float* Rs, float* eta , float* tmp_d ){
     bool iftmp = false;
     if (tmp_d == nullptr){
          init_vec_in_mem_d<float>(tmp_d, n);
          iftmp = true;             
     }               
     get_Gradial(tmp_d, Rij_d, n, Rs, eta);         
     float a = 1.0 ;     
     cublasSaxpy( global_cublasHandle, n, &a, tmp_d, 1, rst_d, 1);
     if (iftmp) clearMemo_d(tmp_d);          
}; 




template <>
void Gfunction_t<double>::get_Gangular_add(double* rst_d, double* Rij_d, double* Rik_d, double* Rjk_d, size_t n, double* eta, double* zeta, double* lambd , double* tmp_d){
     bool iftmp = false;
     if (tmp_d == nullptr){
          init_vec_in_mem_d<double>(tmp_d, n);
          iftmp = true;             
     }               
     get_Gangular(tmp_d, Rij_d, Rik_d, Rij_d, n, eta, zeta, lambd);     
     double a = 1.0 ;     
     cublasDaxpy( global_cublasHandle, n, &a, tmp_d, 1, rst_d, 1);
     if (iftmp) clearMemo_d(tmp_d);          
};



template <>
void Gfunction_t<float>::get_Gangular_add(float* rst_d, float* Rij_d, float* Rik_d, float* Rjk_d, size_t n, float* eta, float* zeta, float* lambd , float* tmp_d ){
     bool iftmp = false;
     if (tmp_d == nullptr){
          init_vec_in_mem_d<float>(tmp_d, n);
          iftmp = true;             
     }               
     get_Gangular(tmp_d, Rij_d, Rik_d, Rij_d, n, eta, zeta, lambd);     
     float a = 1.0 ;     
     cublasSaxpy( global_cublasHandle, n, &a, tmp_d, 1, rst_d, 1);
     if (iftmp) clearMemo_d(tmp_d);          
};













//================================================================================
// tester

int main(int argc, char** argv){ 

     cout << "Usage:  THIS.EXE  DISTANCE_FILE  [-" << FLAG_DISTFILE_HEADLINE << "=1]"
          << "[-" << FLAG_COLUMN_INDEX_FILE  << "=NONE]"  
          << "[-" << FLAG_PARAM_FILE         << "=H_rad|H_ang|O_rad|O_ang]"
          << "[-" << FLAG_ATOM_ORDER_FILE    << "=NONE]"
          << "[-" << FLAG_CUDA_DEVICE_INDEX  << "=0]"
          << endl << endl;

     int device = getCmdLineArgumentInt(argc, (const char **)argv, FLAG_CUDA_DEVICE_INDEX);     
     checkCudaErrors( cudaSetDevice(device) );
     currCuda.update();
     cublas_start();

     Gfunction_t<double> gf;     // the G-function
     
     // distance file headline
     int distheadline = getCmdLineArgumentInt(argc, (const char **)argv, FLAG_DISTFILE_HEADLINE);     
     if(distheadline==0) distheadline=1;     // a special line for test case
          
     
          
     // column index file
     string colidxfile;
     getCmdLineArgumentString(argc, (const char **)argv, FLAG_COLUMN_INDEX_FILE, colidxfile);
     
     
     // parameter file
     string paramfile;
     getCmdLineArgumentString(argc, (const char **)argv, FLAG_PARAM_FILE, paramfile);
     
     
     // atom order file
     string ordfile;
     getCmdLineArgumentString(argc, (const char **)argv, FLAG_ATOM_ORDER_FILE, ordfile);          
          

     // make_G(distfile, distheadline, column_idx file, param file, order file)
     gf.make_G(argv[1], distheadline, colidxfile.c_str(), paramfile.c_str(), ordfile.c_str());
     // resutls saved in gf.G which is a map<string:atom_type, double**>
     
     /*
     
     // Show results
     std::cout.precision(std::numeric_limits<double>::digits10+1);  
     for(auto it= gf.G.begin(); it!=gf.G.end(); it++){
          string atom         = gf.model.atoms[it->first]->name;
          string atom_type    = gf.model.atoms[it->first]->type;
          cout << " G-fn : " << atom << " = " << endl;          
          for(int ii=0; ii<3; ii++){
               for(int jj=0; jj<gf.G_param_max_size[atom_type]; jj++){
                    if ((jj>0)&&( jj%3 ==0 ) ) cout << endl;
                    cout << fixed << setw(16) << it->second[jj][ii] << " " ;                       
               }
          cout << endl;
          }               
     }
     */
    
    
     cublas_end();
    
     return 0;
}





