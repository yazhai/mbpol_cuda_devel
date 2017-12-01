
#include <iostream>

#include <cuda.h>
#include <cublas_v2.h>

#include "utility.h"
#include "utility_cu.cuh"
#include "timestamps.h"
#include "Gfunction_cu.cuh"


using namespace std;

const size_t PITCH_TILE_IN_BITS = 512;









__device__ void testvar_d(double*& dat){
     double a = 135.2;
     dat = &a ; 
}



__global__ void testvar_g(double*& a){
     testvar_d(a);
}






int main1(void){

   //  cudaDeviceReset();

     timers_t timers;
     timerid_t id;
     
     
     checkCudaErrors( cudaSetDevice(1) );
     cublas_start();

     
     
     double** dat_h = nullptr,  ** dst_h=nullptr;
     double*  dat_d = nullptr , * dst_d = nullptr;
     int * idx_h = nullptr;
     
     size_t rows=0, cols=0, pitch=0, pitch_dst=0;
     
        
     //const char* file = "NN_input_2LHO_correctedD6_f64.dat";     
     const char* file = "test.dat";
     
     cout<< " Readin file : " << endl;
     

     timers.insert_random_timer(id,0," Read in");     
     timers.timer_start(id);          
     read2DArray_with_max_thredhold_d(dat_d, pitch, rows, cols, file,1);
     timers.timer_end(id, false, true);
     
     
     init_vec_in_mem_d(dst_d, cols);
     
     timers.insert_random_timer(id,0," Transpose ");     
     timers.timer_start(id);          
     //transpose_mtx_d(dst_d, pitch_dst, dat_d, pitch, rows, cols);
     // The max index vector is stored on host
     //get_max_idx_each_row_d(idx_h, dat_d, pitch, rows, cols);
     
     //norm_rows_in_mtx_by_max_idx_d(dat_d, pitch, rows, cols, idx_h, 4, -2);
     
     //get_Gradial_add(dst_d, dat_d, cols, 0.2, 0.3 );
     
     //get_cos ( dst_d, dat_d, &dat_d[1*pitch/8], &dat_d[2*pitch/8], cols);
     //get_cos_g <<< (cols-1+currCuda.TILEDIMX)/currCuda.TILEDIMX , currCuda.TILEDIMX>>> (dst_d, dat_d, &dat_d[1*pitch/8], &dat_d[2*pitch/8], cols);
     
     
     norm_rows_by_maxabs_in_each_row_d(dat_d, pitch, rows, cols, 4,  6 , 1, 3);
     
//     cutoff_g<<<1, 5>>>(dst_d, dat_d, 5);

     init_mtx_in_mem<double> (dat_h, 5, 3);

     init_mtx_in_mem<double> (dst_h, 15, 9);
     
     std::cout << " dat_h address is " << dat_h << std::endl;
     
     std::cout << " dst_h address is " << dst_h << std::endl;

     std::cout << " dat_d address is " << dat_d << std::endl;
     
     std::cout << " dst_d address is " << dst_d << std::endl;
     
     double *a=nullptr ;
    // testvar_g<<<1, 1>>>(a);
     std::cout << " a's address is " << a << std::endl;

     //std::cout << " a's value is " << *a << std::endl;

     
     timers.timer_end(id, false, true);    
     
     
     printDeviceMatrix(dat_d, pitch, rows, 7);
     
     printDeviceVector(dst_d, cols);
     
     
     
     
     
     
     
     
     
     
     
     //printDeviceMatrix(dst_d, pitch_dst, cols, rows);
     //printDeviceVector(idx_d, rows);


    // for (int i=0; i<rows; i++){
    //      cout << idx_h[i] << " " ;
    // }
     cout << endl;
     
     clearMemo(dat_h);
     clearMemo(dst_h);
     
     cout << " Here " << endl;
     clearMemo_d(dat_d);
     cout << " Here " << endl;
     clearMemo_d(dst_d);
     cout << " Here " << endl;
     timers.get_all_timers_info();
    // delete[] idx_h;
     
     cublas_end();     
     return 0;
}
