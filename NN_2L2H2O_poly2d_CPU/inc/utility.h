#ifndef UTILITY_H
#define UTILITY_H



#include <limits>
#include <string.h>

#include "atomTypeID.h"

//==============================================================================
// A 2D-array type based on vector
template<typename T>
using matrix_by_vector_t = std::vector<std::vector<T> >;




//==============================================================================
//
// Check if a string is a float number
template <typename T>
bool IsFloat( std::string myString );


//==============================================================================
//
// Memory management functions
//
template <typename T>
void clearMemo(T** & data);

template <typename T>
void clearMemo(std::vector<T**> & data);


template <typename T>
void clearMemo(std::map<std::string, T**> & data);



//==============================================================================
//
// Initialize a matrix in consecutive memory
template <typename T>
bool init_mtx_in_mem(T** & data, size_t rows, size_t cols);



//==============================================================================
//
// Read in a 2D array from file and save to  **data / rows / cols
template <typename T>
int read2DArrayfile(T** & data, size_t& rows, size_t& cols, const char* file, int titleline=0);


template <typename T>
int read2DArray_with_max_thredhold(T** & data, size_t& rows, size_t& cols, const char* file, int titleline=0,
    int thredhold_col=0, T thredhold_max=std::numeric_limits<T>::max());




//========================================================================================
// 2D array transpose
//
template <typename T>
void transpose_mtx(T** & datdst,  T** datrsc,  size_t nrow_rsc, size_t ncol_rsc);



#if defined (_USE_GSL) || defined (_USE_MKL)
// Using cblas_dcopy and cblas_scopy if cblas libraries are employed
template <>
void transpose_mtx<double>(double** & datdst, double** datrsc,  size_t nrow_rsc, size_t ncol_rsc);

template <>
void transpose_mtx<float>(float** & datdst,  float** datrsc, size_t nrow_rsc, size_t ncol_rsc);

#endif


//===============================================================================                                     
//
// Matrix normalization utility functions
size_t get_count_by_percent(size_t src_count, double percentage);
                                     
                                     
template<typename T>
void get_max_each_row(T*& rst,  T* src,  size_t src_rows, size_t src_cols, long int col_start=0, long int col_end=-1);


#if defined (_USE_GSL) || defined (_USE_MKL)

template<>
void get_max_each_row<double>(double*& rst, double* src, size_t src_rows, size_t src_cols, long int col_start, long int col_end);

template<>
void get_max_each_row<float>(float*& rst, float* src, size_t src_rows, size_t src_cols, long int col_start, long int col_end);
#endif


template<typename T>
void norm_rows_in_mtx_by_col_vector(T* src_mtx, size_t src_rows, size_t src_cols, T* scale_vec, 
            long int col_start=0, long int col_end=-1 );



#if defined (_USE_GSL) || defined (_USE_MKL)
template<>
void norm_rows_in_mtx_by_col_vector<double>(double* src_mtx, size_t src_rows, size_t src_cols, double* scale_vec, long int col_start, long int col_end);

template<>
void norm_rows_in_mtx_by_col_vector<float>(float* src_mtx, size_t src_rows, size_t src_cols, float* scale_vec, long int col_start, long int col_end);
#endif


template<typename T>
void norm_rows_by_maxabs_in_each_row(T* src_mtx, size_t src_rows, size_t src_cols, long int max_start_col=0, long int max_end_col=-1, long int norm_start_col =0, long int norm_end_col=-1);






//==============================================================================
//
// Utility functions dealing with input arguments
//
int stringRemoveDelimiter(char delimiter, const char *string);

bool checkCmdLineFlag(const int argc, const char **argv, const char *string_ref);

int getCmdLineArgumentInt(const int argc, const char **argv, const char *string_ref);

bool getCmdLineArgumentString(const int argc, const char **argv,
                                     const char *string_ref, std::string & string_retval);
                                                                        



//allowed types
extern template bool init_mtx_in_mem<unsigned int>(unsigned int ** & data, size_t rows, size_t cols);

extern template bool IsFloat<double>( std::string myString );
extern template bool IsFloat<float>( std::string myString );

extern template void clearMemo<double> (double ** & data);
extern template void clearMemo<float>(float ** & data);
extern template void clearMemo<unsigned int>(unsigned int ** & data);

extern template void clearMemo<double>(std::vector<double**> & data);
extern template void clearMemo<float>(std::vector<float**> & data);
extern template void clearMemo<unsigned int>(std::vector<unsigned int**> & data);

extern template void clearMemo<double>(std::map<std::string, double**> & data);
extern template void clearMemo<float>(std::map<std::string, float**> & data);
extern template void clearMemo<unsigned int>(std::map<std::string, unsigned int**> & data);

extern template int read2DArrayfile<double>(double** & data, size_t& rows, size_t& cols, const char* file, int titleline=0);
extern template int read2DArrayfile<float>(float** & data, size_t& rows, size_t& cols, const char* file, int titleline=0);

extern template int read2DArray_with_max_thredhold<double>(double** & data, size_t& rows, size_t& cols, const char* file, int titleline, 
                                            int thredhold_col, double thredhold_max);
extern template int read2DArray_with_max_thredhold<float>(float ** & data, size_t& rows, size_t& cols, const char* file, int titleline, 
                                            int thredhold_col, float thredhold_max);

extern template void transpose_mtx<double>(double** & datdst,  double** datrsc,  size_t nrow_rsc, size_t ncol_rsc);
extern template void transpose_mtx<float>(float** & datdst,  float** datrsc,  size_t nrow_rsc, size_t ncol_rsc);

extern template void get_max_each_row<double>(double *& rst,  double* src,  size_t src_rows, size_t src_cols, 
                        long int col_start, long int col_end); 
extern template void get_max_each_row<float>(  float *& rst,   float* src,  size_t src_rows, size_t src_cols, 
                        long int col_start, long int col_end); 

extern template void norm_rows_in_mtx_by_col_vector<double>(double* src_mtx, size_t src_rows, size_t src_cols, double* scale_vec, 
                        long int col_start=0, long int col_end=-1 );
extern template void norm_rows_in_mtx_by_col_vector<float>(float* src_mtx, size_t src_rows, size_t src_cols, float* scale_vec, 
                        long int col_start=0, long int col_end=-1 );

extern template void norm_rows_by_maxabs_in_each_row<double>(double* src_mtx, size_t src_rows, size_t src_cols, long int max_start_col, 
        long int max_end_col, long int norm_start_col , long int norm_end_col);
extern template void norm_rows_by_maxabs_in_each_row<float>(float* src_mtx, size_t src_rows, size_t src_cols, long int max_start_col, 
        long int max_end_col, long int norm_start_col , long int norm_end_col);
#endif
