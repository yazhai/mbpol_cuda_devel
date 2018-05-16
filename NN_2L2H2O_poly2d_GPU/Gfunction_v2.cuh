#ifndef GFUNCTION_V2_H
#define GFUNCTION_V2_H


#include <cstdlib>
#include <vector>
#include <string>
#include <map>
#include <limits>

#include "readGparams_v2.h"
#include "atomTypeID_v2.h"
#include "timestamps.h"
#include "utility_cu.cuh"
#include "utility.h"

#include<cuda.h>
#include<cublas_v2.h>


const int COL_RAD_RS = 3;
const int COL_RAD_ETA = 2; 
const int COL_RAD_TOTAL = 5;

const int COL_ANG_ETA = 2;
const int COL_ANG_ZETA = 4;
const int COL_ANG_LAMBD=3;
const int COL_ANG_TOTAL=6;

#define MAXPARAM 384      //maximum num parameters for an atom sequence (or arbitrary large number for shared memory instantiation),max(np*nc)

//=================================// DEVICE FUNCTIONS //=================================//

//get cosine between two vectors ij, and ik
template<typename T>
__device__ T get_cos_d(T Rij_d, T Rik_d, T Rjk_d) {

     T Rijxik = Rij_d*Rik_d;  

     if ( Rijxik != 0 ) {
          return  ( ( Rij_d*Rij_d + Rik_d*Rik_d - Rjk_d*Rjk_d )/ (2.0 * Rijxik) );
     } else {
          //rst_d =  std::numeric_limits<T>::infinity();
          return  -98765.4321;   // some strange value
     }
}


//calculate distance between 2 xyz positions
/* Inputs Example:  xyz1/2: formatted: x1,x2,x3,x4,y1,y2,y3,y4,z1,z2,z3,z4
                    N: num clusters(4 in ex.)
                    id: index to calculate (ex: 3 = distance(xyz1(x3y3z3), xyz2(x3,y3,z3)) )
*/                  
template<typename T>
__device__ inline T get_dist(const T * xyz1, const T * xyz2, const size_t N, const size_t id){
     T a = xyz1[id] - xyz2[id];
     T b = xyz1[N+id] - xyz2[N+id];
     T c = xyz1[2*N+id] - xyz2[2*N+id];
     return sqrt(a*a + b*b + c*c);
}

//base case for switching function. used in 2b/3b cutoff calculations
template<typename T>
__device__ T base_fswitch(T ri,T rf,T r){
     T value;
     T coef = M_PI/(rf-ri);
     T temp = (1.0 + cos(coef*(r-ri)))/2.0;
     if(r<rf)
          value = (r>=ri)? temp : 1;
     else     
          value = 0;

     return value;
} 

// Get G_radial
template<typename T>
__device__ T get_Gradial_d(T Rij_d, T Rs, T eta){
     T G_rad = exp( -eta * ( (Rij_d-Rs)*(Rij_d-Rs) )  )  ;
     return G_rad;
};

// Get G_angular
template<typename T>
__device__ T get_Gangular_d(T Rij_d, T Rik_d, T Rjk_d, T eta, T zeta, T lambd){         
     T  G_ang =   2 * pow( (1.0 + lambd* get_cos_d(Rij_d, Rik_d, Rjk_d))/2.0, zeta) 
                         * exp(-eta*  ( (Rij_d+Rik_d+Rjk_d)*(Rij_d+Rik_d+Rjk_d) ) );    
     
     return G_ang ;    
};


//=================================// GLOBAL FUNCTIONS //=================================//

//Function get_Gradial:  get the Radial results for a specific sequence in all clusters 
/*   Inputs:   g:        device pointer to g function where output is stored
 *             pitch:    pitch of matrix, g, on device
 *             xyz0:     pointer to xyz of atom 0. Stored as xxxyyyzzz
 *             xyz1:     pointer to xyz of atom 1. Stored as xxxyyyzzz
 *             p:        pointer to the relevant matrix of parameters for this sequence
 *             np:       number of parameters
 *             nc:       number of columns in parameter matrix
 *             N:        number of clusters
 *             offset:   current offset of where to store results in g at the end of calculation
 *   Alg:      load params into shared memory, calculate distance, calculate result.      
 *   Result:   A specific section of g, starting at offset, for each cluster is filled
 *                       with the correct radial results.
 */     
template<typename T>
__global__ void get_Gradial(T * g, size_t pitch, T * xyz0, T * xyz1, T * p, size_t np, size_t nc, size_t N, size_t offset){	
	
     __shared__ T params[MAXPARAM];

     int tid = threadIdx.x + blockDim.x * blockIdx.x;
     int stride = blockDim.x * gridDim.x;

     if(threadIdx.x<np*nc){
          params[threadIdx.x] = p[threadIdx.x];
     }
     __syncthreads();

     while(tid < N){

          T distance = get_dist<T> (xyz0, xyz1, N, tid);

          for(int ip = 0; ip < np; ip++){
               g[(offset+ip)*pitch/sizeof(T) + tid] += get_Gradial_d(distance, params[ip*nc + COL_RAD_RS], params[ip*nc + COL_RAD_ETA]);
          }
          tid += stride;
     }
}

//Function get_Gradial2:  get the Radial results for a specific sequence in all clusters, and store in 2 output matrices 
//This function operates in exactly the same way as get_Gradial, but stores results in 2 g matrices instead of one.
//The purpose of this function is to create a special case and avoid computing the same results twice
template<typename T>
__global__ void get_Gradial2(T * g0, T * g1, size_t pitch,  T * xyz0, T * xyz1, T * p, size_t np, size_t nc, size_t N, size_t offset){	
	
     __shared__ T params[MAXPARAM];
     

     int tid = threadIdx.x + blockDim.x * blockIdx.x;
     int stride = blockDim.x * gridDim.x;
     T temp = 0;

     if(threadIdx.x<np*nc){
          params[threadIdx.x] = p[threadIdx.x];

     }

     __syncthreads();

     while(tid < N){
   
          T distance = get_dist<T> (xyz0, xyz1, N, tid);
	
		for(int ip =0; ip<np; ip++){
               temp = get_Gradial_d(distance, params[ip*nc + COL_RAD_RS], params[ip*nc + COL_RAD_ETA]);
               g0[(offset+ip)*pitch/sizeof(T) + tid] += temp;
               g1[(offset+ip)*pitch/sizeof(T) + tid] += temp;
		}

          tid += stride;
     }
}

//Function get_Gangular:  get the Angular results for a specific sequence in all clusters 
//This function operates in the same way as the radial function, but calls the angular device function instead.
template<typename T>
__global__ void get_Gangular(T * g, size_t pitch, T * xyz0, T * xyz1, T * xyz2, T * p, size_t np, size_t nc, size_t N, size_t offset){	

     __shared__ T params[MAXPARAM];
    
     int tid = threadIdx.x + blockDim.x * blockIdx.x;
     int stride = blockDim.x * gridDim.x;

     if(threadIdx.x<np*nc){
          params[threadIdx.x] = p[threadIdx.x];

     }
     __syncthreads();

     while(tid < N){
      
          T distance1 = get_dist<T> (xyz0, xyz1,N,tid);
          T distance2 = get_dist<T> (xyz0, xyz2,N,tid);
          T distance3 = get_dist<T> (xyz1, xyz2,N,tid);
   
		for (int ip = 0; ip < np; ip++){  
               g[(offset+ip)*pitch/sizeof(T) + tid] += get_Gangular_d(distance1, distance2, distance3, params[ip*nc + COL_ANG_ETA], 
                                                        params[ip*nc + COL_ANG_ZETA], params[ip*nc + COL_ANG_LAMBD]) ;
		}

          tid += stride;
     }

}

//switching function - 2 body
template <typename T>
__global__ void fswitch_2b(T * cutoffs,T * xyz0, T * xyz1, T ri, T rf, size_t N) {
     int tid = threadIdx.x + blockDim.x * blockIdx.x;
     int stride = blockDim.x * gridDim.x;
     T dist;

     while(tid < N){
          dist = get_dist<T> (xyz0, xyz1, N, tid);  //distance between 2 oxygen atoms
          cutoffs[tid] = base_fswitch(ri,rf,dist);
          tid += stride;
    }

}

//switching function - 3 body
template <typename T>
__global__ void fswitch_3b(T * cutoffs, T * xyz0, T * xyz1, T * xyz2, T ri, T rf, size_t N) {
     int tid = threadIdx.x + blockDim.x * blockIdx.x;
     int stride = blockDim.x * gridDim.x;
     T s01,s02,s12;

     while(tid < N){
          s01 = base_fswitch(ri,rf,get_dist<T> (xyz0, xyz1, N, tid));  //distance between 2 oxygen atoms
          s02 = base_fswitch(ri,rf,get_dist<T> (xyz0, xyz2, N, tid));
          s12 = base_fswitch(ri,rf,get_dist<T> (xyz1, xyz2, N, tid));
     
          cutoffs[tid] = s01*s02 + s01*s12 + s02*s12;
          tid += stride;
     }
}

//THIS FUNCTION IS STILL VERY UNOPTIMAL.
//steps to improve: Share parameter memory, and coalesce data. 
template <typename T>
__global__ void scale(size_t N, size_t paramSize, T * scaleVec, T * dst, size_t pitch){
     int tid = threadIdx.x + blockDim.x * blockIdx.x;
     int stride = blockDim.x*gridDim.x;
     int stride2 = pitch/sizeof(T);
     while(tid < N){
          for(int i = 0; i< paramSize; i++){
               dst[i*stride2 + tid] /= scaleVec[i];
          }
          tid+= stride;     
     }

}


//G FUNCTION CLASS
template <typename T>
class Gfunction_t{
private:
timers_t timers;
timerid_t id;
	 
//==============================================================================================
//
public:


// Variables:
atom_Type_ID_t<T> model;         // Model, save information about atom names, indexes, and types
size_t NCluster, NType;                 // Number of dimer/trimer in the model / 
idx_t* TypeStart;      // The starting atom index of one type  
idx_t* TypeNAtom;      // The count of the atoms in one type

T** xyz;                    //xyz data of atoms
matrix_2D_d<T> * xyz_d;      //xyz data of atoms stored in device memory (xxxxyyyyzzzz)
T* cutoffs;                 //switching function values (Nsamples long) -- for use after NN

Gparams_t<T> gparams;                 // G-fn paramter class

std::vector<T**> G;   // G-fn matrix
std::vector < matrix_2D_d<T> *> G_d;         //stored as param x N matrix 
std::vector<size_t> G_param_max_size;        //max size of parameters for each atom type



Gfunction_t();

~Gfunction_t();

void load_xyz(const char* file);

void load_paramfile(const char* file);

void load_paramfile_default();

// load sequnece file
void load_seq(const char* _seqfile);

void init_G();

void load_cutoffs();

void scale_G(const char ** _scaleFiles);

void make_G_XYZ(const char* _xyzFile, const char * _paramfile, const char* _ordfile, const char ** _scaleFiles);

void make_G();

};

extern template class Gfunction_t<double>;
extern template class Gfunction_t<float>;


#endif
