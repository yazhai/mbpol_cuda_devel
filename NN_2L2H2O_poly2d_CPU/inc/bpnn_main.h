#ifndef BPNNPLUGIN_H
#define BPNNPLUGIN_H


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

#include "readGparams_v2.h"
#include "atomTypeID_v2.h"
#include "utility.h"
#include "timestamps.h"
#include "Gfunction_v2.h"
#include "network.h"

#define INFILE_3B   "22.hdf5" 
#define INFILE_2B   "34.hdf5"
#define INFILE1     INFILE_3B    // HDF5 files for Layer Data


#define HMAX_FILE  "H_MAX"
#define OMAX_FILE  "O_MAX"

//possible last chars of weight data(used to differentiate between weights and biases)
#define CHECKCHAR1  "W"                 // dense_1_[W]           for "W"
#define CHECKCHAR2  "l"                 // dense_1/kerne[l]      for "l"

//percentage of gfn output data used in training(and not in testing)
#define TRAIN_PERCENT 0

#define EUNIT (6.0)/(.0433634) //energy conversion factor 
//#define EUNIT 1

//define openMP
#ifdef _OPENMP
#include <omp.h>
#endif 

// const char* FLAG_COLUMN_INDEX_FILE =   "columnfile" ;
// const char* FLAG_PARAM_FILE        =    "paramfile" ;
// const char* FLAG_ATOM_ORDER_FILE   =      "ordfile" ;
// const char* FLAG_GFN_OUTPUT_ENABLE =          "gfnOut";     //enable intermediate gfn output to file
#define FLAG_COLUMN_INDEX_FILE    "columnfile" ;
#define FLAG_PARAM_FILE            "paramfile" ;
#define FLAG_ATOM_ORDER_FILE         "ordfile" ;
#define FLAG_GFN_OUTPUT_ENABLE        "gfnOut" ;   

const int THREDHOLD_COL = -1;
const double THREDHOLD_MAX_VALUE = 60.0;


namespace MBbpnnPlugin{



template<typename T>
T get_eng_2h2o(size_t nd, std::vector<T>xyz1, std::vector<T>xyz2, std::vector<std::string> atoms1 = std::vector<std::string>() , std::vector<std::string> atoms2 = std::vector<std::string>() );

template<typename T>
T get_eng_2h2o(size_t nd, std::vector<T>xyz1, std::vector<T>xyz2, std::vector<T> & grad1, std::vector<T>& grad2, std::vector<std::string> atoms1 = std::vector<std::string>() , std::vector<std::string> atoms2 = std::vector<std::string>() );



// a temperary function for performance testing
template<typename T>
T get_eng_2h2o(const char* xyzfile, bool ifgrad=false);

template<typename T>
T get_eng_3h2o(const char* xyzfile, bool ifgrad=false);




template<typename T>
class bpnn_t : public MBbpnnPlugin::Gfunction_t<T>, public MBbpnnPlugin::allNN_t<T>{


public:
     bpnn_t ();
     bpnn_t (std::string tag) ;
     ~bpnn_t();


     T* energy_ ; 


}; // end of bpnn_t class




};  // end of namespace

#endif