#ifndef UTILITY_CUH
#define UTILITY_CUH



#include <limits>
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

#include<cuda.h>
#include<cublas_v2.h>

#include "atomTypeID.h"
#include "utility.h"

#ifdef _OPENMP
#include <omp.h>
#endif 

//==============================================================================
//
// Some functions are from NVIDIA's open source in this file.
//
/**
* Copyright 2014 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/
#define FatalError(s) {                                                \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;\
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    exit(EXIT_FAILURE);                                                \
}

#define checkCUDNN(status) {                                           \
    std::stringstream _error;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      _error << "CUDNN failure\nError: " << cudnnGetErrorString(status); \
      FatalError(_error.str());                                        \
    }                                                                  \
}

#define checkCudaErrors(status) {                                      \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure\nError: " << cudaGetErrorString(status); \
      FatalError(_error.str());                                        \
    }                                                                  \
}

#define checkCublasErrors(status) {                                    \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cublas failure\nError code " << status;        \
      FatalError(_error.str());                                        \
    }                                                                  \
}



//==============================================================================
// 
// General setting on GPU kernels
#ifndef INST_GLOBAL_VAR
extern float  FCONST [2];
extern double DCONST [2];
#endif

struct cudaInfo{
private:
     int dev;     
     
     
public:
     cudaInfo();
     ~cudaInfo();
     void update();
     
     size_t sharedMemPerBlock;
     
     int  maxThreadsPerBlock;
     int  maxThreadsDim[3];
     int  maxGridSize[3];    
     int  warpSize ;         
     
     size_t TILEDIMX, TILEDIMY; 
};

extern cudaInfo currCuda;  // A global class setting up for CUDA kernels.



//==============================================================================
// 
// cublas start/stop utilities
extern cublasHandle_t global_cublasHandle;  // global handle

void cublas_start();
void cublas_end();



//==============================================================================
//
// Check if a type is single precision floating point or double precision floating point
template <typename T>
struct TypeIsFloat
{
     static const bool value = false;
};

template <>
struct TypeIsFloat<float>
{    
     static const bool value = true;
};

template <typename T>
struct TypeIsDouble
{
     static const bool value = false;
};

template <>
struct TypeIsDouble<double>
{    
     static const bool value = true;
};



//==============================================================================
//
// Device Memory release
//
template <typename T>
void clearMemo_d(T* & data){
     if (data != NULL)
     {
       checkCudaErrors( cudaFree(data) );            
     }
     return;
};


template <typename T>
void clearMemo_d(std::vector<T*> & data){
     for (auto it=data.rbegin(); it!=data.rend(); it++){
          clearMemo_d<T>(*it);
          data.pop_back();
     }
     return;
}


template <typename T>
void clearMemo_d(std::map<std::string, T*> & data){
     for (auto it=data.begin(); it!=data.end(); it++){
          clearMemo_d<T>(it->second);
          it->second = nullptr;
     }
     return;
}





// Pinned memory release
template <typename T>
void clearMemo_p(T* & data){
     if (data != NULL)
     {
       checkCudaErrors( cudaFreeHost(data) );            
     }
     return;
};


template <typename T>
void clearMemo_p(std::vector<T*> & data){
     for (auto it=data.rbegin(); it!=data.rend(); it++){
          clearMemo_p<T>(*it);
          data.pop_back();
     }
     return;
}


template <typename T>
void clearMemo_p(std::map<std::string, T*> & data){
     for (auto it=data.begin(); it!=data.end(); it++){
          clearMemo_p<T>(it->second);
          it->second = nullptr;
     }
     return;
}




//==============================================================================
//
// Initialize a vector in consecutive memory on device
template <typename T>
bool init_vec_in_mem_d(T* & data, size_t size){
     clearMemo_d(data);
     checkCudaErrors( cudaMalloc( (void**) &data, size*sizeof(T)   ) );
     checkCudaErrors( cudaMemset(data, 0, size*sizeof(T)) ); 
     return true;
};



// Initialize a matrix in consecutive memory on device
template <typename T>
bool init_mtx_in_mem_d(T* & data, size_t & pitch, size_t nrows, size_t ncols){
     clearMemo_d(data);     
     size_t T_s = sizeof(T);
     checkCudaErrors( cudaMallocPitch(&data, &(pitch), ncols*T_s, nrows) );
     checkCudaErrors( cudaMemset(data, 0, pitch*nrows) ); 
     return true;
};



// Pinned memories are initialized via cudaMallocHost
// and is accessable by both device and host
// Initialize a vector in consecutive pinned memory
template <typename T>
bool init_vec_in_mem_p(T* & data, size_t size){
     clearMemo_p(data);
     checkCudaErrors( cudaMallocHost( (void**) &data, size*sizeof(T)   ) );
     checkCudaErrors( cudaMemset(data, 0, size*sizeof(T)) ); 
     return true;
};


//==============================================================================
//
// Helper function showing the data vector on Device
template <typename T>
void printDeviceVector(T* vec_d, size_t size)
{
    T *vec;
    vec = new T[size];
    cudaDeviceSynchronize();
    cudaMemcpy(vec, vec_d, size*sizeof(T), cudaMemcpyDeviceToHost);
    if(TypeIsDouble<T>::value) {
          std::cout.precision(std::numeric_limits<double>::digits10+1);
    } else {
          std::cout.precision(std::numeric_limits<float>::digits10+1);;
    }
    std::cout.setf( std::ios::fixed, std::ios::floatfield );
    for (int i = 0; i < size; i++)
    {
        std::cout << (vec[i]) << " ";
    }
    std::cout << std::endl;
    delete [] vec;
}


// Helper function showing the data matrix on Device
// Showing the first rowsXcols number of elements.
template <typename T>
void printDeviceMatrix(T* mtx_d, size_t pitch, size_t rows, size_t cols)
{
    T *vec;
    vec = new T[rows*cols];
    size_t T_s = sizeof(T);    
    cudaDeviceSynchronize();
    cudaMemcpy2D((void*)vec, T_s*cols, (const void*)mtx_d, pitch, T_s*cols, rows, cudaMemcpyDeviceToHost);
    if(TypeIsDouble<T>::value) {
          std::cout.precision(std::numeric_limits<double>::digits10+1);
    } else {
          std::cout.precision(std::numeric_limits<float>::digits10+1);;
    }
    std::cout.setf( std::ios::fixed, std::ios::floatfield );    
    for (int i = 0; i < rows ; i++){
          for( int j=0; j < cols ; j++){
               std::cout << (vec[i*cols + j]) << " ";   
          }
          std::cout << std::endl;
    }
    std::cout << std::endl;
    delete [] vec;
}



//==============================================================================
//
// Vector / Matrix copy functions
//
// Vector copy from host to device
template <typename T>
void memcpy_vec_h2d(T* & data_d, T* data_h, size_t size)
{
     if (data_d == NULL)
     {
          init_vec_in_mem_d<T>(data_d, size);
     }        
    
    int size_b = size*sizeof(T);
    checkCudaErrors( cudaMemcpy((void*)data_d, (const void*)data_h,
                                size_b,
                                cudaMemcpyHostToDevice) );
}   ;


// Vector copy from device to host
template <typename T>
void memcpy_vec_d2h(T* & data_h, T* data_d, size_t size)
{
    if (data_h == NULL)
    {
          data_h = new T[size];
    }            
    int size_b = size*sizeof(T);
    checkCudaErrors( cudaMemcpy((void*)data_h, (const void*)data_d,
                                size_b,
                                cudaMemcpyDeviceToHost) );                            
}   ;

// Matrix copy from host to device
// Note matrix in device is represented by T*
// but matrix in host is represented by T**
template <typename T>
void memcpy_mtx_h2d(T* & data_d, size_t & pitch_d, T** data_h, size_t rows, size_t cols)
{
    if (data_d == NULL)
    {
         init_mtx_in_mem_d<T>(data_d, pitch_d, rows, cols);
    }        
    size_t T_s = sizeof(T);        
    checkCudaErrors( cudaMemcpy2D( (void*)data_d, pitch_d, (const void*)(*data_h), T_s*cols, 
                                T_s*cols,  rows,
                                cudaMemcpyHostToDevice) );
}   ;


// Matrix copy from device to host
// Note matrix in device is represented by T*
// but matrix in host is represented by T**
template <typename T>
void memcpy_mtx_d2h(T** & data_h, T* data_d, size_t pitch_d,  size_t rows, size_t cols)
{
    if ( (data_h == NULL) || (*data_h == NULL) )
    {
         init_mtx_in_mem<T>(data_h, rows, cols);
    }        
    size_t T_s = sizeof(T);        
    checkCudaErrors( cudaMemcpy2D( (void*)(*data_h), T_s*cols,  (const void*)data_d,  pitch_d, 
                                T_s*cols,  rows,
                                cudaMemcpyDeviceToHost) );
}   ;




//========================================================================================
// 2D array transpose
//
// Some nasty way of transpose matrix on device . Perhaps need optimization here!!!
template <typename T>
__global__ void trans_d(T* dat_dst, size_t pitch_dst, T* dat_rsc, size_t pitch_rsc)
{
     size_t T_s = sizeof(T);
     int idx_i = blockIdx.x *  (int)(pitch_rsc/T_s) + threadIdx.x;
     int idx_o = threadIdx.x * (int)(pitch_dst/T_s) + blockIdx.x;  
     dat_dst[idx_o] = dat_rsc[idx_i];
};


template <typename T>
void transpose_mtx_d(T* & dat_dst, size_t & pitch_dst, T* dat_rsc, size_t pitch_rsc, size_t nrow_rsc, size_t ncol_rsc)
{
     if (dat_dst == NULL){
          init_mtx_in_mem_d<T>(dat_dst, pitch_dst, ncol_rsc, nrow_rsc);          
     }     
     std::cout << " Matrix transpose of non-float/non-double type is called. " << std::endl;
     std::cout << " Note this action is not fully implemented. " << std::endl;
     trans_d<T><<<nrow_rsc, ncol_rsc>>>(dat_dst, pitch_dst, dat_rsc, pitch_rsc);       
};

template <>
void transpose_mtx_d<float>(float* & dat_dst, size_t & pitch_dst, float* dat_rsc, size_t pitch_rsc, size_t nrow_rsc, size_t ncol_rsc);

template <>
void transpose_mtx_d<double>(double* & dat_dst, size_t & pitch_dst, double* dat_rsc, size_t pitch_rsc, size_t nrow_rsc, size_t ncol_rsc);


//==============================================================================
//
// Read in a 2D array from file and save to  *data / rows / cols
template <typename T>
int read2DArray_d(T* & data, size_t & pitch, size_t & rows, size_t & cols, const char* file, int titleline=0){
    try {           
          clearMemo_d<T>(data);          
          std::ifstream ifs(file);
          std::string line;
          matrix_by_vector_t<T> mtx;
          
          for (int i=0; i < titleline; i++){          
               getline(ifs,line);
          }
          std::vector<T> onelinedata;
          while(getline(ifs, line)){          
               char* p = &line[0u];
               char* end;              
               onelinedata.clear();                              
               for( T d = strtod(p, &end); p != end; d = strtod(p, &end) ) {
                    p = end;                    
                    onelinedata.push_back(d);           
               };               
               if(onelinedata.size()>0) mtx.push_back(onelinedata);              
          }          

          rows=mtx.size();
          if (rows > 0){   
               cols=mtx[0].size();
               size_t T_s = sizeof(T);
               
               init_mtx_in_mem_d<T>(data, pitch, rows, cols);  
                         
               //#ifdef _OPENMP
               //#pragma omp parallel for simd shared(data, mtx, rows)
               //#endif                                      
               for(int ii=0; ii<rows; ii++){
                    checkCudaErrors( cudaMemcpy(    (void*) ( &data[ii*(size_t)(pitch/T_s)] ), 
                                   (const void*)(&mtx[ii][0]),
                                   cols*T_s,
                                   cudaMemcpyHostToDevice) );  
               }           
          } else {
               std::cout << " No Data is read from file as 2D array" << std::endl;
          }                          
          mtx.clear();
          return 0;                    
    } catch (const std::exception& e) {
        std::cerr << " ** Error ** : " << e.what() << std::endl;
        return 1;
    }
};


template <typename T>
int read2DArray_with_max_thredhold_d(T* & data, size_t & pitch, size_t& rows, size_t& cols, const char* file, int titleline=0, int thredhold_col=0, T thredhold_max=std::numeric_limits<T>::max()){
    try { 
          
          clearMemo_d<T>(data);
          std::ifstream ifs(file);
          std::string line;
          matrix_by_vector_t<T> mtx;
          
          for (int i=0; i < titleline; i++){          
               getline(ifs,line);
          }
          std::vector<T> onelinedata;
          while(getline(ifs, line)){          
               char* p = &line[0u];
               char* end;              
               onelinedata.clear();                              
               for( T d = strtod(p, &end); p != end; d = strtod(p, &end) ) {
                    p = end;                    
                    onelinedata.push_back(d);           
               };               
               if (onelinedata.size()>0) {   
                    
                    int checkcol = onelinedata.size() ;                    
                    if ( thredhold_col >=0) {           
                         // when thredhold_index is non-negative, check the colum VS max
                         checkcol = thredhold_col;
                    } else {
                         // when thredhold_index is negative, check the column from the end VS max
                         checkcol += thredhold_col;                                         
                    }                                
                    if (onelinedata[checkcol] > thredhold_max) {
                         continue;   // If the data exceeds thredhold, ignore this line.
                    }
                    
                    mtx.push_back(onelinedata);                               
               }                           
          }          
          rows=mtx.size();
          if (rows > 0){     
               cols=mtx[0].size();          
               size_t T_s = sizeof(T);
               init_mtx_in_mem_d<T>(data, pitch, rows, cols);  

               //#ifdef _OPENMP
               //#pragma omp parallel for simd shared(data, mtx, rows)
               //#endif                                      
               for(int ii=0; ii<rows; ii++){
                    checkCudaErrors( cudaMemcpy(    (void*) ( &data[ii*(size_t)(pitch/T_s)] ), 
                                   (const void*)(&mtx[ii][0]),
                                   cols*T_s,
                                   cudaMemcpyHostToDevice) );  
               }                     
          } else {
               std::cout << " No Data is read from file as 2D array" << std::endl;
          }                    
          mtx.clear();
          return 0;                    
    } catch (const std::exception& e) {
        std::cerr << " ** Error ** : " << e.what() << std::endl;
        return 1;
    }
};



//===============================================================================                                     
//
// Matrix normalization utility functions
template<typename T>
void get_max_idx_each_row_d(int*& rst_idx,  T* src, size_t pitch, size_t src_rows, size_t src_cols, long int col_start=0, long int col_end=-1){ 
          //if(col_end < 0) col_end = src_cols + col_end ;  // change negative column index to positive        
          //if(rst_idx == nullptr) init_vec_in_mem_d(rst_idx, src_rows);   
          //size_t T_s = sizeof(T);   
                   
          std::cout << " The function get_max_idx_each_row_d for non-float/non-double type is not implemented" << std::endl;                   
          //#ifdef _OPENMP
          //#pragma omp parallel for simd shared(src, rst, src_rows, src_cols, col_start, col_end)
          //#endif   
          //for(size_t ii=0 ; ii< src_rows ; ii++ ){
          //     for(size_t jj=col_start; jj<= col_end ; jj++){
          //          if ( rst[ii] < abs(src[ii*(int)(pitch/T_s) + jj]) ) {
          //               rst[ii] = abs(src[ii*(int)(pitch/T_s) + jj]);
          //          };
          //     }
          //};                         
};


template<>
void get_max_idx_each_row_d<float>(int*& rst_idx,  float* src, size_t pitch, size_t src_rows, size_t src_cols, long int col_start, long int col_end);

template<>
void get_max_idx_each_row_d<double>(int*& rst_idx,  double* src, size_t pitch, size_t src_rows, size_t src_cols, long int col_start, long int col_end);






template<typename T>
void norm_rows_in_mtx_by_max_idx_d(T*& src_mtx, size_t pitch, size_t src_rows, size_t src_cols, int* scale_idx_in_each_row_h, long int col_start=0, long int col_end=-1 ){
     //if(col_end < 0) col_end = src_cols + col_end ;  // change negative column index to positive
     //// scale each row (from offset index) in a matrix by a column vector
     //#ifdef _OPENMP
     //#pragma omp parallel for simd shared(src_mtx, src_rows, src_cols, scale_vec, col_start, col_end)
     //#endif
     //for(int i = 0; i< src_rows; i++){     
     //     T scale = 1 / scale_vec[i];
     //     for(int j=col_start; j<= col_end; j++){
     //          src_mtx[src_cols*i + j] = src_mtx[src_cols*i + j] * scale ;
     //     }
     //}
     std::cout << " The function norm_rows_in_mtx_by_max_idx_d for non-float/non-double type is not implemented" << std::endl;
}



template<>
void norm_rows_in_mtx_by_max_idx_d<double>(double*& src_mtx, size_t pitch, size_t src_rows, size_t src_cols, int* scale_idx_in_each_row_h, long int col_start, long int col_end);

template<>
void norm_rows_in_mtx_by_max_idx_d<float>(float*& src_mtx, size_t pitch, size_t src_rows, size_t src_cols, int* scale_idx_in_each_row_h , long int col_start, long int col_end);



template<typename T>
void norm_rows_by_maxabs_in_each_row_d(T*& src_mtx, size_t pitch, size_t src_rows, size_t src_cols, long int max_start_col=0, long int max_end_col=-1, long int norm_start_col =0, long int norm_end_col=-1){     
     
     int* norm_idx_h = new int[src_rows]();     
     
     get_max_idx_each_row_d<T>(norm_idx_h, src_mtx, pitch, src_rows, src_cols, max_start_col, max_end_col);   
     norm_rows_in_mtx_by_max_idx_d<T>(src_mtx, pitch, src_rows, src_cols, norm_idx_h, norm_start_col, norm_end_col);          
     delete[] norm_idx_h;
     
}



//==============================================================================
//
// CUDA Utility Helper Functions
//
static void  showDevices( void );


#endif
