#ifndef GFUNCTION_CUH
#define GFUNCTION_CUH

#include <cstdlib>
#include <vector>
#include <string>
#include <map>
#include <limits>

#include "readGparams_cu.cuh"
#include "atomTypeID.h"
#include "timestamps.h"
#include "utility.h"
#include "utility_cu.cuh"

#ifdef _OPENMP
#include <omp.h>
#endif 

#include<cuda.h>
#include<cublas_v2.h>


const int COL_RAD_RS = 3;
const int COL_RAD_ETA = 2; 
const int COL_ANG_ETA = 2;
const int COL_ANG_ZETA = 4;
const int COL_ANG_LAMBD=3;


//===========================================================================================
// __device__ functions for matrix elementary calculation
//
// 
//  Cutoff
template<typename T>
__device__ T cutoff_d(T Rij_d, T R_cut = 10) {         
     if ( Rij_d <= R_cut ) {              // Needs benchmarking to check if putting the branch here is better or putting it at the end.        
          T t = 1.0 - Rij_d/(R_cut) ;      
          t = exp(2*t);
          t = (t-1) / (t+1);                
          return t * t * t ;    
     }  else {
          return 0.0 ;
     };
};


// Get cosine
template<typename T>
__device__ T get_cos_d(T Rij_d, T Rik_d, T Rjk_d) {
     //cosine of the angle between two vectors ij and ik    
     T Rijxik = Rij_d*Rik_d;    
     if ( Rijxik != 0 ) {
          return  ( ( Rij_d*Rij_d + Rik_d*Rik_d - Rjk_d*Rjk_d )/ (2.0 * Rijxik) );
     } else {
          //rst_d =  std::numeric_limits<T>::infinity();
          return  -98765.4321;   // some strange value
     }
};



// Get G_radial
template<typename T>
__device__ T get_Gradial_d(T Rij_d, T Rs, T eta, T R_cut=10.0){
     T G_rad = cutoff_d(Rij_d, R_cut);     
     if ( G_rad > 0 ) {
          G_rad *= exp( -eta * ( (Rij_d-Rs)*(Rij_d-Rs) )  )  ;
     }
     return G_rad;
};

// Get G_angular
template<typename T>
__device__ T get_Gangular_d(T Rij_d, T Rik_d, T Rjk_d, T eta, T zeta, T lambd){    
    T G_ang = cutoff_d(Rij_d)*cutoff_d(Rik_d)*cutoff_d(Rjk_d);    
    if ( G_ang > 0) {    
          G_ang *=   2 * pow( (1.0 + lambd* get_cos_d(Rij_d, Rik_d, Rjk_d))/2.0, zeta) 
                     * exp(-eta*  ( (Rij_d+Rik_d+Rjk_d)*(Rij_d+Rik_d+Rjk_d) ) );    
    } 
    return G_ang ;    
};



//===========================================================================================
// __global__ vectorized functions
// calling from host to device
// 
// Vectorized functions list:
//void cutoff_g(T* Rdst, T* Rrsc, size_t n, T R_cut=10);
//void get_cos_g(T * Rdst, T * Rij, T * Rik, T * Rjk, size_t n);
//void get_Gradial_g(T* Rdst, T* Rij, size_t n, T Rs, T eta, T R_cut=10);
//void get_Gangular_g(T* Rdst, T* Rij, T* Rik, T* Rjk, size_t n, T eta,T zeta, T lambd );
//void cum_add_g(T* rst_d, T* src_d, size_t n);
template<typename T>
__global__ void cutoff_g(T* rst_d, T* Rij_d, size_t n, T R_cut = 10) {    
     int idx = blockIdx.x *  blockDim.x + threadIdx.x;
     if ( idx < n ){        
          rst_d[idx] = cutoff_d( Rij_d[idx], R_cut);                     
     };
};


template<typename T>
__global__ void get_cos_g(T* rst_d, T* Rij_d, T* Rik_d, T* Rjk_d, size_t n) {
     int idx = blockIdx.x *  blockDim.x + threadIdx.x;
     if ( idx < n ){     
          rst_d[idx] = get_cos_d ( Rij_d[idx], Rik_d[idx], Rjk_d[idx] );
     }
};


template<typename T>
__global__ void get_Gradial_g(T* rst_d, T* Rij_d, size_t n, T* Rs, T* eta, T R_cut=10 ){ 
  int idx = blockIdx.x *  blockDim.x + threadIdx.x;
  if (idx < n) {
     rst_d[idx] = get_Gradial_d( Rij_d[idx] , *Rs, *eta, R_cut);
  };
};


template<typename T>
__global__ void get_Gangular_g(T* rst_d, T* Rij_d, T* Rik_d, T* Rjk_d, size_t n, T* eta, T* zeta, T* lambd ){
     int idx = blockIdx.x *  blockDim.x + threadIdx.x;
     if (idx < n) {
          rst_d[idx]=get_Gangular_d(Rij_d[idx], Rik_d[idx], Rjk_d[idx], *eta, *zeta, *lambd);
     };
};


template<typename T>
__global__ void cum_add_g(T* rst_d, T* src_d, size_t n){
     int idx = blockIdx.x *  blockDim.x + threadIdx.x;
     if (idx < n) {
          rst_d[idx] += src_d[idx] ;   
     };
};





// some tester functions
template <typename T>
void cutoff(T*  Rdst, T*  Rrsc, size_t n, T R_cut=10){
     cutoff_g <<< (n-1+currCuda.TILEDIMX)/currCuda.TILEDIMX , currCuda.TILEDIMX >>> (Rdst, Rrsc, n, R_cut);
};

template <typename T>
void get_cos(T *  Rdst, T *  Rij, T *  Rik, T *  Rjk, size_t n){
     get_cos_g <<< (n-1+currCuda.TILEDIMX)/currCuda.TILEDIMX , currCuda.TILEDIMX >>> (Rdst, Rij, Rik, Rij, n);
};





//===========================================================================================
//
// Class Gfunction 
//
template <typename T>
class Gfunction_t{
private:

// 
// These methods are launched on multiple kernels 
//
void get_Gradial(T* rst_d, T* Rij_d, size_t n, T* Rs, T* eta, T R_cut=10 ){ 
     get_Gradial_g <<< (n-1+currCuda.TILEDIMX)/currCuda.TILEDIMX , currCuda.TILEDIMX >>> (rst_d, Rij_d, n, Rs, eta, R_cut);
     cudaThreadSynchronize();
};


void get_Gradial_add(T* rst_d, T* Rij_d, size_t n, T* Rs, T* eta , T* tmp_d = nullptr ){
     bool iftmp = false;
     if (tmp_d == nullptr){
          init_vec_in_mem_d<T>(tmp_d, n);
          iftmp = true;             
     }               
     get_Gradial(tmp_d, Rij_d, n, Rs, eta);        
     cum_add_g <<< (n-1+currCuda.TILEDIMX)/currCuda.TILEDIMX , currCuda.TILEDIMX >>> (rst_d, tmp_d, n);
     cudaThreadSynchronize();
     if (iftmp) clearMemo_d(tmp_d);          
}; 


void get_Gangular(T* rst_d, T* Rij_d, T* Rik_d, T* Rjk_d, size_t n, T* eta, T* zeta, T* lambd ){
     get_Gangular_g <<< (n-1+currCuda.TILEDIMX)/currCuda.TILEDIMX , currCuda.TILEDIMX >>> (rst_d, Rij_d, Rik_d, Rjk_d, n, eta, zeta, lambd);
     cudaThreadSynchronize();
};


void get_Gangular_add(T* rst_d, T* Rij_d, T* Rik_d, T* Rjk_d, size_t n, T* eta, T* zeta, T* lambd, T* tmp_d = nullptr ){
     bool iftmp = false;
     if (tmp_d == nullptr){
          init_vec_in_mem_d<T>(tmp_d, n);
          iftmp = true;             
     }               
     get_Gangular(tmp_d, Rij_d, Rik_d, Rij_d, n, eta, zeta, lambd);        
     cum_add_g <<< (n-1+currCuda.TILEDIMX)/currCuda.TILEDIMX , currCuda.TILEDIMX >>> (rst_d, tmp_d, n);
     cudaThreadSynchronize();
     if (iftmp) clearMemo_d(tmp_d);          
};


// Timers for benchmarking
timers_t timers;
timerid_t id, id1, id2 , id3;



//==============================================================================================
//
public:


// Nested class storing G_fn data
struct G_mtx_t{
     size_t    nrow, ncol,pitch,nrow_max;
     T*        mtx_d ;

     G_mtx_t(): mtx_d(nullptr), nrow(0), ncol(0), pitch(0) , nrow_max(0) {}; // Default constructor
     G_mtx_t(size_t _nrow, size_t _ncol) {   // init with nrow/ncol
          init_G_mtx(_nrow, _ncol);
     };              
     ~G_mtx_t(){
          clearMemo_d(mtx_d);     
     };           
          
     void init_G_mtx(size_t _nrow, size_t _ncol) {
          nrow = _nrow;
          ncol = _ncol;      
          init_mtx_in_mem_d( mtx_d, pitch, nrow, ncol);
          nrow_max = (size_t) pitch/sizeof(T);               
     };     
     void assign_mtx( T* _imtx_d , size_t _pitch, size_t _nrow, size_t _ncol){
          mtx_d = _imtx_d;
          pitch = _pitch;
          nrow = _nrow;
          ncol = _ncol;
          nrow_max = (size_t) pitch/sizeof(T);        
     };     
     
     
     T* get_gmtx_dptr(size_t irow, size_t icol){
          return &(mtx_d[irow*nrow_max+icol]) ;
     
     }
};




// Variables:
atom_Type_ID_t model;         // Model, save information about atom names, indexes, and types
size_t natom;                 // Number of atoms registered in the model
idx_t* colidx_p;              // distance matrix column index mapping, saved as pinned (accessable by both host and device)

T* distT_d;                   // distance matrix transpose on device [number of measured distances X nunmber of dimers]
size_t ndimers, ndistcols;    // number of dimers[ncol in distT_d], number of measured distances [nrow in distT_d]
size_t distT_pitch;           // pitch size of dist/T matrix





Gparams_t<T> GP;                 // G-fn paramter class
std::map<std::string, std::map<idx_t, int> > G_param_start_idx;   // Index of each parameter in G-fn
std::map<std::string, std::map<idx_t, int> > G_param_size;        // Parameter sizes in G-fn
std::map<std::string, size_t>                G_param_max_size;    // Total size of G-fn


std::map<idx_t, G_mtx_t*> G;   // G-fn matrix


// Gfunction class constructor/destructor
//

Gfunction_t(){
     colidx_p = nullptr;
     distT_d = nullptr;
};


~Gfunction_t(){
     clearMemo_p(colidx_p);
     clearMemo_d<T>(distT_d);     
     for(auto it=G.begin() ; it!=G.end(); it++){
          delete it->second;
     };
};





//==============================================================================================
//
// load distance matrix column index 
//
//
void load_dist_colidx(const char* _dist_idx_file){        // not implemented yet
     if( strlen(_dist_idx_file)>0 ) {     
          std::cout << " Loading custom distance matrix colum index is not implemented yet !" << std::endl;         
          load_dist_colidx_default_d();     
     }  else {
          load_dist_colidx_default_d();
     }
};    

void load_dist_colidx_default_d(){ 
     idx_t** colidx_h = nullptr;
     model.load_default_atom_id(colidx_h, natom); // 
     init_vec_in_mem_p(colidx_p, natom*natom);
     checkCudaErrors( cudaMemcpy( (void*)colidx_p, (const void*) colidx_h[0], natom*natom*sizeof(idx_t) , cudaMemcpyHostToHost) );
     clearMemo(colidx_h);                       
};


idx_t* get_colidx_ptr (idx_t atom1, idx_t atom2){  
     return &(colidx_p[atom1*natom + atom2]) ;
}



// load distance matrix and filt out samples that exceed a maximum value
void load_distfile(const char* _distfile, int _titleline=0, int _thredhold_col=0, T thredhold_max=std::numeric_limits<T>::max()){
     timers.insert_random_timer( id, 0, "Read_distance_file");
     timers.timer_start(id);
     T* dist_tmp_d=nullptr;
     size_t pitch_tmp;
     
     if (thredhold_max < std::numeric_limits<T>::max() ) {
          int ifread = read2DArray_with_max_thredhold_d(dist_tmp_d, pitch_tmp, ndimers, ndistcols, _distfile, _titleline, _thredhold_col, thredhold_max);     
     } else {     
          int ifread = read2DArray_d(dist_tmp_d, pitch_tmp , ndimers, ndistcols, _distfile, _titleline);
     }
     transpose_mtx_d<T>(distT_d, distT_pitch, dist_tmp_d, pitch_tmp, ndimers, ndistcols);          
     clearMemo_d(dist_tmp_d);    
     timers.timer_end(id);
     
    //std::cout << " READ in count of dimers = " << ndimers << std::endl;
};


T* get_distT_dptr(size_t irow, size_t icol){
     return &(distT_d[irow * (distT_pitch/sizeof(T))+ icol] );
};








// load parameter matrix 

void load_paramfile(const char* _paramfile){
     if ( strlen(_paramfile) > 0 ) {     
          GP.read_param_from_file(_paramfile, model); 
     } else {
          load_paramfile_default();
     }     
     GP.updateParam_d();    
};


void load_paramfile_default(){  
     GP.read_param_from_file("H_rad", model); 
     GP.read_param_from_file("H_ang", model);
     GP.read_param_from_file("O_rad", model);
     GP.read_param_from_file("O_ang", model);  
}



// load sequnece file

void load_seq(const char* _seqfile){
     if ( strlen(_seqfile) >0 ){
          GP.read_seq_from_file(_seqfile, model);
     } else {
          GP.make_seq_default();
     }
};




//=================================================================================
// G-function Construction
//void make_G(const char* _distfile, int _titleline, const char* _colidxfile, const char* _paramfile, const char* _ordfile);
//void make_G();
void make_G(const char* _distfile, int _titleline, const char* _colidxfile, const char* _paramfile, const char* _ordfile, int _thredhold_col=0, T thredhold_max=std::numeric_limits<T>::max()){     

     load_distfile(_distfile, _titleline, _thredhold_col, thredhold_max);     
     load_dist_colidx(_colidxfile);
     load_paramfile(_paramfile);
     load_seq(_ordfile);         
     make_G();
};


void make_G(){      
     timers.insert_random_timer(id3, 1 , "Gf_run_all");
     timers.timer_start(id3);     
             
     for(auto it = GP.seq.begin(); it!=GP.seq.end(); it++) {
          // it->first  = name of atom type ;
          // it->second = vector<idx_t> ;   a vector saving the list of sequence order
          int curr_idx = 0;
          for(auto it2 = it->second.begin(); it2!=it->second.end(); it2++){                
               // *it2 = element relationship index (=atom1_type * atom2_type * ... )
               G_param_start_idx[it->first][*it2]     = curr_idx;      // The start index is the cumulative size of all relationships until now                                                          
               size_t _tmp = GP.params_d[it->first][*it2]->nparam;
               G_param_size[it->first][*it2] = _tmp;
               curr_idx +=  _tmp;        // next relatinoship               
               //std::cout << it->first << " " << *it2 << " " << curr_idx << std::endl;
          }          
          G_param_max_size[it->first] = curr_idx;                // max capacity of this atom type
     }
          
     // For each atom1
     //#pragma omp parallel for shared(model, GP, natom, colidx, distT, ndimers, G, G_param_start_idx, G_param_max_size)
     for (auto at1=model.atoms.begin(); at1!=model.atoms.end(); at1++ ) {
          
          std::string atom1 = at1->second->name;                   
          idx_t idx_atom1 = at1->first;
          std::string atom1_type = at1->second->type;
          idx_t idx_atom1_type = model.types[atom1_type]->id; 

          //std::cout << " Dealing with atom : " << atom1 << " ... " << std::endl;          
          
          if( G_param_max_size.find(atom1_type) != G_param_max_size.end() ){
               
               
               G_mtx_t* g = new G_mtx_t;
               g->init_G_mtx(G_param_max_size[atom1_type] , ndimers) ;
               
               T* tmp_d = nullptr ;
               init_vec_in_mem_d<T>(tmp_d, ndimers);  // a temporary space for cache

               
               timers.insert_random_timer(id2, 2 , "Gfn_rad+ang_all");
               timers.timer_start(id2);            
               
               // for each atom2
               for(auto at2= model.atoms.begin(); at2!=model.atoms.end(); at2++ ){
                    std::string atom2 = at2->second->name;
                    if(atom1 != atom2){
                         
                         idx_t idx_atom2 = at2->first;
                         std::string atom2_type = at2->second->type;
                         idx_t idx_atom2_type = model.types[atom2_type]->id;
                         idx_t idx_atom12 = idx_atom1_type*idx_atom2_type;

                         // Calculate RAD when it is needed
                         if ( G_param_start_idx[atom1_type].find(idx_atom12) != G_param_start_idx[atom1_type].end() ) {                     
                              std::cout << atom1 << " - " << atom2 << std::endl;
                              size_t nrow_params =  G_param_size[atom1_type][idx_atom12];
                              idx_t* icol = get_colidx_ptr(idx_atom1, idx_atom2);  // col index of the distance to retrieve                              
                              
                              T* Rs=nullptr, * eta= nullptr;                         
                              int idx_g_atom12 = G_param_start_idx[atom1_type][idx_atom12];
                              
                              for(size_t i=0 ; i< nrow_params; i++){       
                                   Rs   =  GP.params_d[atom1_type][idx_atom12]->get_param_dptr(i, COL_RAD_RS) ;
                                   eta  =  GP.params_d[atom1_type][idx_atom12]->get_param_dptr(i, COL_RAD_ETA);                                                                    
                                   timers.insert_random_timer(id, idx_atom12, "GRadial");
                                   timers.timer_start(id);

                                   // ico is the column index saved on host/device; Rs, eta are parameters saved on device;
                                   // Memory on device can NOT be called out on a HOST function, even as parameters!!!!           
                                   // So here every params is changed to either ptr or pinned memory
                                   get_Gradial_add(  g->get_gmtx_dptr(idx_g_atom12+i ,0) , get_distT_dptr(*icol, 0) , ndimers, Rs, eta, tmp_d);          

                                   timers.timer_end(id);
                              }   
                         }


                         
                         timers.insert_random_timer(id1, 3, "Gfn_ang_all");
                         timers.timer_start(id1);                      
                         
                         for(auto at3=next(at2,1) ; at3!=model.atoms.end(); at3++){
                              std::string atom3 = at3->second->name;
                              if(atom3 != atom1) {
                                   idx_t idx_atom3 = at3->first;
                                   std::string atom3_type = at3->second->type;
                                   idx_t idx_atom3_type = model.types[atom3_type]->id;
                                   idx_t idx_atom123 = idx_atom12*idx_atom3_type;

                                   if( G_param_start_idx[atom1_type].find(idx_atom123) != G_param_start_idx[atom1_type].end() ) {
                                   
                                        std::cout << atom1 << " - " << atom2 << " - " << atom3 << std::endl;                      
                                        idx_t* icol  = get_colidx_ptr(idx_atom1, idx_atom2) ; // col index of the distance to retrieve
                                        idx_t* icol2 = get_colidx_ptr(idx_atom1, idx_atom3) ; // col index of the distance to retrieve
                                        idx_t* icol3 = get_colidx_ptr(idx_atom2, idx_atom3) ; // col index of the distance to retrieve
                                        size_t nrow_params =  GP.params_d[atom1_type][idx_atom123]->nparam;                              
                                        
                                        T* lambd=nullptr, *zeta=nullptr, *eta=nullptr;
                                        int idx_g_atom123 = G_param_start_idx[atom1_type][idx_atom123];

                                        for(size_t i=0 ; i< nrow_params; i++){      
                                             lambd = ( GP.params_d[atom1_type][idx_atom123]->get_param_dptr(i, COL_ANG_LAMBD) ); 
                                             eta   = ( GP.params_d[atom1_type][idx_atom123]->get_param_dptr(i, COL_ANG_ETA  ) ); 
                                             zeta  = ( GP.params_d[atom1_type][idx_atom123]->get_param_dptr(i, COL_ANG_ZETA ) );                    
                                             timers.insert_random_timer(id, idx_atom123, "GAngular");
                                             timers.timer_start(id);
                                             get_Gangular_add(g->get_gmtx_dptr(idx_g_atom123+i, 0) , get_distT_dptr(*icol,0) , get_distT_dptr(*icol2,0),  get_distT_dptr(*icol3,0), ndimers, eta, zeta, lambd, tmp_d);         
                                             timers.timer_end(id, true, false);        
                                        } 
                                   } 

                              }                         
                         }
                         
                         timers.timer_end(id1);
                    }     
               }
               clearMemo_d(tmp_d);          
               //delete[] tmp;
               
               // Save results to G-fn
               G[at1->first] = g;

               timers.timer_end(id2);                                       
          }
     }
     
     
     timers.timer_end(id3);
     timers.get_all_timers_info();
     timers.get_time_collections();
}









};



template <>
void Gfunction_t<double>::get_Gradial_add(double* rst_d, double* Rij_d, size_t n, double* Rs, double* eta, double* tmp_d );

template <>
void Gfunction_t<float>::get_Gradial_add(float* rst_d, float* Rij_d, size_t n, float* Rs, float* eta , float* tmp_d );

template <>
void Gfunction_t<double>::get_Gangular_add(double* rst_d, double* Rij_d, double* Rik_d, double* Rjk_d, size_t n, double* eta, double* zeta, double* lambd , double* tmp_d);

template <>
void Gfunction_t<float>::get_Gangular_add(float* rst_d, float* Rij_d, float* Rik_d, float* Rjk_d, size_t n, float* eta, float* zeta, float* lambd , float* tmp_d );


#endif
