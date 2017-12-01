#include <cstdlib>
#include <iomanip>
#include <utility>
#include <cstddef>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include <iostream>
#include <sstream>
#include <fstream>

#include <vector>
#include <map>
#include <string>

#define INST_GLOBAL_VAR
#include "utility_cu.cuh"
#include "atomTypeID.h"

#ifdef _OPENMP
#include <omp.h>
#endif 


#include<cuda.h>
#include<cublas_v2.h>


#define FLAGSTART '-'
#define FLAGASSGN '='


using namespace std;

__constant__ float  FCONST [2];
__constant__ double DCONST [2];


//==============================================================================
// 
// General setting on GPU kernels
cudaInfo currCuda;

cudaInfo::cudaInfo(){
     update();
     float   fconst[2] ;
     double  dconst[2] ;
     fconst[0] = 0.0;
     fconst[1] = 1.0;
     dconst[0] = 0.0;
     dconst[1] = 1.0;     
     checkCudaErrors( cudaMemcpyToSymbol( (const void*)FCONST, (const void*) fconst, 2* sizeof(float), 0, cudaMemcpyHostToDevice ) );
     checkCudaErrors( cudaMemcpyToSymbol( (const void*)DCONST, (const void*) dconst, 2*sizeof(double), 0, cudaMemcpyHostToDevice ) );
};

cudaInfo::~cudaInfo(){};

void cudaInfo::update(){
     checkCudaErrors( cudaGetDevice(&dev) );
     cudaDeviceProp devProp;
     checkCudaErrors( cudaGetDeviceProperties(&devProp, dev) );
     sharedMemPerBlock = devProp.sharedMemPerBlock;
     warpSize = devProp.warpSize;
     maxThreadsPerBlock = devProp.maxThreadsPerBlock ;
     for(int i=0; i<3; i++){
          maxThreadsDim[i] =  devProp.maxThreadsDim[i];
          maxGridSize[i]   =  devProp.maxGridSize[i];
     } ;     
     
     
     // A default setting on tile size;
     TILEDIMX = warpSize * 8 ;         // Setting threads per block in X = 32 * 8 = 256
     TILEDIMY = maxThreadsPerBlock / TILEDIMX ;  
     if (TILEDIMY > maxThreadsDim[1]) TILEDIMY = maxThreadsDim[1] ; 
     
};





//==============================================================================
// 
// cublas start/stop utilities
cublasHandle_t global_cublasHandle;  // global handle

void cublas_start(){
     checkCublasErrors( cublasCreate(&global_cublasHandle) );
};

void cublas_end(){
     checkCublasErrors( cublasDestroy(global_cublasHandle) ) ;
};



//==============================================================================
//
// Matrix transpose
//template <typename T>
//void transpose_mtx_d(T* & dat_dst, size_t & pitch_dst, T* dat_rsc, size_t pitch_rsc, size_t nrow_rsc, size_t ncol_rsc)

// Function definition for a specific type should be left in cpp file
template <>
void transpose_mtx_d<float>(float* & dat_dst, size_t & pitch_dst, float* dat_rsc, size_t pitch_rsc, size_t nrow_rsc, size_t ncol_rsc){
     //float a = 1, b=0;     
     if( dat_dst == NULL) init_mtx_in_mem_d( dat_dst, pitch_dst, ncol_rsc, nrow_rsc ) ;
     int T_s = sizeof(float);
     checkCublasErrors(  cublasSgeam ( global_cublasHandle,     CUBLAS_OP_T,   CUBLAS_OP_T,
                                        nrow_rsc, ncol_rsc,
                                        (const float*) &FCONST[1],
                                        (const float*) dat_rsc,  (int) (pitch_rsc/T_s),
                                        (const float*) &FCONST[0],
                                        (const float*) dat_rsc,  (int) (pitch_rsc/T_s),
                                                       dat_dst,  (int) (pitch_dst/T_s)) );                                                       
};

template <>
void transpose_mtx_d<double>(double* & dat_dst, size_t & pitch_dst, double* dat_rsc, size_t pitch_rsc, size_t nrow_rsc, size_t ncol_rsc){
     //double a = 1, b=0;     
     if( dat_dst == NULL) init_mtx_in_mem_d( dat_dst, pitch_dst, ncol_rsc, nrow_rsc ) ;
     int T_s = sizeof(double);
     checkCublasErrors(  cublasDgeam ( global_cublasHandle,     CUBLAS_OP_T,   CUBLAS_OP_T,
                                        nrow_rsc, ncol_rsc,
                                        (const double*) &DCONST[1],
                                        (const double*) dat_rsc,  (int) (pitch_rsc/T_s),
                                        (const double*) &DCONST[0],
                                        (const double*) dat_rsc,  (int) (pitch_rsc/T_s),
                                                       dat_dst,  (int) (pitch_dst/T_s) ) );                                        
};


//===============================================================================                                     
//
// Matrix normalization utility functions     
//
template<>
void get_max_idx_each_row_d<double>(int*& rst_idx_h, double* src, size_t pitch, size_t src_rows, size_t src_cols, long int col_start, long int col_end){
     if(col_start < 0) col_start = src_cols + col_start;
     if(col_end < 0) col_end = src_cols + col_end ;  // change negative column index to positive
     if(rst_idx_h == nullptr) rst_idx_h = new int[src_rows];     
     size_t T_s = sizeof(double);     
     //#ifdef _OPENMP
     //#pragma omp parallel for simd shared(src, rst, src_rows, src_cols, col_start, col_end)
     //#endif   
     // Note rst_idx is a vector stored in host. Its returned value from function starts from 1, not 0; so that we need some adjustment     
     for(size_t ii=0 ; ii< src_rows ; ii++ ){
          int jj = (int)(pitch/T_s)*ii + col_start ;   
          checkCublasErrors( cublasIdamax( global_cublasHandle, 
                                   (col_end - col_start + 1), 
                                   (&src[jj]) , 1, (&rst_idx_h[ii])) );                                                                                                             
          if (rst_idx_h[ii] ==0 ) {
               std::cout << " Finding max_abs element is unsuccessful ... " << endl;                                                                      
          } else {                                                           
               rst_idx_h[ii] += col_start - 1; 
          }
     };         
};


template<>
void get_max_idx_each_row_d<float>(int*& rst_idx_h, float* src, size_t pitch, size_t src_rows, size_t src_cols, long int col_start, long int col_end){
     if(col_start < 0) col_start = src_cols + col_start;
     if(col_end < 0) col_end = src_cols + col_end ;  // change negative column index to positive
     if(rst_idx_h == nullptr) rst_idx_h = new int[src_rows];            
     size_t T_s = sizeof(float);
     //#ifdef _OPENMP
     //#pragma omp parallel for simd shared(src, rst, src_rows, src_cols, col_start, col_end)
     //#endif   
     // Note rst_idx is a vector stored in host. Its returned value from function starts from 1, not 0; so that we need some adjustment 
     for(size_t ii=0 ; ii< src_rows ; ii++ ){
          int jj = (int)(pitch/T_s)*ii + col_start ;          
          checkCublasErrors( cublasIsamax( global_cublasHandle, 
                                   (col_end - col_start + 1), 
                                   (&src[jj]) , 1, (&rst_idx_h[ii])) );     
          if (rst_idx_h[ii] ==0 ) {
               std::cout << " Finding max_abs element is unsuccessful ... " << endl;                                                                      
          } else {
               rst_idx_h[ii] += col_start - 1;                          
          }
     }; 
};


template<>
void norm_rows_in_mtx_by_max_idx_d<double>(double*& src_mtx, size_t pitch, size_t src_rows, size_t src_cols, int* scale_idx_in_each_row, long int col_start, long int col_end){
     if(col_start < 0) col_start = src_cols + col_start;
     if(col_end < 0) col_end = src_cols + col_end ;  // change negative column index to positive
     
     size_t T_s = sizeof(double);
     size_t pitch_cols = pitch/T_s;
     
     for (size_t i=0; i< src_rows; i++){
          double scale =0 ;                             
          checkCudaErrors( cudaMemcpy( &scale, (const void*) &(src_mtx[i*pitch_cols + scale_idx_in_each_row[i]]) , T_s, cudaMemcpyDeviceToHost ) );          
          scale = 1.0/scale;
          checkCublasErrors( cublasDscal(global_cublasHandle,    (int) col_end - col_start +1,
                                   (const double*) &scale,   (double*)&(src_mtx[i*pitch_cols + col_start]), 1)  );     }
};

template<>
void norm_rows_in_mtx_by_max_idx_d<float>(float*& src_mtx, size_t pitch, size_t src_rows, size_t src_cols, int* scale_idx_in_each_row , long int col_start, long int col_end){
     if(col_start < 0) col_start = src_cols + col_start;
     if(col_end < 0) col_end = src_cols + col_end ;  // change negative column index to positive
     
     size_t T_s = sizeof(float);
     size_t pitch_cols = pitch/T_s;
     
     for (size_t i=0; i< src_rows; i++){
          float scale =0 ;                   
          checkCudaErrors( cudaMemcpy( &scale, (const void*) &(src_mtx[i*pitch_cols + scale_idx_in_each_row[i]]) , T_s, cudaMemcpyDeviceToHost ) );          
          scale = 1.0/scale;
          checkCublasErrors( cublasSscal(global_cublasHandle,    (int) col_end - col_start +1,
                                   (const float*) &scale,   (float*)&(src_mtx[i*pitch_cols + col_start]), 1)  );     }
};





//==============================================================================
//
// CUDA Utility Helper Functions
//
static void  showDevices( void ){
    int totalDevices;
    checkCudaErrors(cudaGetDeviceCount( &totalDevices ));
    printf("\nThere are %d CUDA capable devices on your machine :\n", totalDevices);
    for (int i=0; i< totalDevices; i++) {
          struct cudaDeviceProp devProp;
          checkCudaErrors(cudaGetDeviceProperties( &devProp, i ));
          printf( "device %d : sms %2d  Capabilities %d.%d, SmClock %.1f Mhz, MemSize (Mb) %d, MemClock %.1f Mhz, Ecc=%d, boardGroupID=%d\n",
                    i, devProp.multiProcessorCount, devProp.major, devProp.minor,
                    (float)devProp.clockRate*1e-3,
                    (int)(devProp.totalGlobalMem/(1024*1024)),
                    (float)devProp.memoryClockRate*1e-3,
                    devProp.ECCEnabled,
                    devProp.multiGpuBoardGroupID);
          printf("Major revision number:         %d\n",  devProp.major);
          printf("Minor revision number:         %d\n",  devProp.minor);
          printf("Name:                          %s\n",  devProp.name);
          printf("Total global memory:           %u\n",  devProp.totalGlobalMem);
          printf("Total shared memory per block: %u\n",  devProp.sharedMemPerBlock);
          printf("Total registers per block:     %d\n",  devProp.regsPerBlock);
          printf("Warp size:                     %d\n",  devProp.warpSize);
          printf("Maximum memory pitch:          %u\n",  devProp.memPitch);
          printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
          for (int i = 0; i < 3; ++i)
          printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
          for (int i = 0; i < 3; ++i)
          printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
          printf("Clock rate:                    %d\n",  devProp.clockRate);
          printf("Total constant memory:         %u\n",  devProp.totalConstMem);
          printf("Texture alignment:             %u\n",  devProp.textureAlignment);
          printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
          printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
          printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));                    
                    
    }
};
