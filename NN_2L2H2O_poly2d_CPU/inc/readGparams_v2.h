#ifndef GPARAMS_CUH
#define GPARAMS_CUH

#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <iomanip>

// #include "atomTypeID_v2.h"
#include "utility.h"



namespace MBbpnnPlugin{

//#ifndef CONST_VAR_INSTANT 
//const int SYMBOL_COUNT = 4096;
//extern __constant__ double PARAMS[SYMBOL_COUNT] ; // constant memory in CUDA. Reserve 32K.
//#endif

// Read parameters for G-function construction
template <typename T>
struct Gparams_t {
public:

     struct param_t {
          std::vector< T > dat_h;
          idx_t nparam, ncol;           
          param_t();
          ~param_t();                 
     };
     
     
     
     // member variables
     std::map< std::string, param_t> PARAMS;  // save read-in parameters
     
     
     // constructor/destructor
     Gparams_t();
     ~Gparams_t(); 
   
     // Read in params from a file; save numbers in `params`; save related atom information into `model`.
     int read_param_from_file(const char* _file);            
} ;

}; // end of namespace MBbpnnPlugin

#endif
