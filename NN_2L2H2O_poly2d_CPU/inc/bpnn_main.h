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

#define INFILE_2B   "34.hdf5"
#define INFILE_3B   "22.hdf5"
#define INFILE1     INFILE_3B    // HDF5 files for Layer Data

#define PARAM_FILE_2B "Gfunc_params_2Bv14_tuned.dat"
#define PARAM_FILE_3B "Gfunc_params_3Bv16.dat"


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

// File as External Varible
extern char inputFile[BUFSIZ]; // moduar
extern char paramFile[BUFSIZ]; // parameter file for G function
extern char orderFile[BUFSIZ]; // order file for G function
extern char scaleFile[BUFSIZ]; // scale file for G function
extern char nnFile[BUFSIZ];    // parameter file for Neural Network


#define STR_USAGE_LONG "\n"\
  "Usage: %s [--input=]XYZ_NAME [--body=2] [--gradient=0] [--param] [--order] "\
  "[--scale] [--nn] [--help]\n"\
  "Calcualte Molecule Enery.\n"\
  "  -i, --input <filename>  \n"\
  "     Name of input file. Usually ends with '.xyz'. Defaults to TODO \n"\
  "  -b, --body <2|3>  \n"\
  "     Specify dimer (default) or trimer \n"\
  "  -g, --gradient <0|1>  \n"\
  "     Specify calculate gradient or not (default) \n"\
  "  -p, --param <filename>  \n"\
  "     Name of parameter file for G function. \n"\
  "  -o, --order <filename>  \n"\
  "     [NOT USED FOR NOW] Name of order file for G function. \n"\
  "  -s, --scale <filename>  \n"\
  "     [NOT USED FOR NOW] Name of scale file for G function. \n"\
  "  -n, --nn <filename>  \n"\
  "     Name of parameter file for Neural Network. Should be in hdf5 format\n"\
  "  -h, --help \n"\
  "     Print the long help string\n"\
  "\n"

#define STR_USAGE_SHORT "\nUsage: %s [--input=]FILENAME [--body] [--gradient] [--param]"\
                          " [--order] [--scale] [--nn] [--help]\n"\
                          "Try '%s -h' for more information.\n"\
                          "Try '%s $(cat my_config)'to run with a configure "\
                          "file. \n"\
                          "\n"

#define OPEN_FILE_ERROR "ERROR: cannot open file:%s \n"


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
