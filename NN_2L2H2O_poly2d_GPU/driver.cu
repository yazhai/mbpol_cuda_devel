
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


#include "Gfunction_v2.cuh"
#include "network.cuh"
#include "utility.h"
#include<cuda.h>
#include<cudnn.h>


#define INFILE_2B   "/server-home1/ndanande/Documents/mbpol_cuda_devel/NN_2L2H2O_poly2d_GPU/34.hdf5"
#define INFILE1     INFILE_2B    // HDF5 files for Layer Data
#define CHECKCHAR "l"                 // dense_1/kerne[l]      for "l"

//scaling
#define SCALE_FILE_H  "/server-home1/ndanande/Documents/mbpol_cuda_devel/NN_2L2H2O_poly2d_GPU/H_max"
#define SCALE_FILE_O  "/server-home1/ndanande/Documents/mbpol_cuda_devel/NN_2L2H2O_poly2d_GPU/O_max"


const char* FLAG_COLUMN_INDEX_FILE =   "columnfile" ;
const char* FLAG_PARAM_FILE        =    "paramfile" ;
const char* FLAG_ATOM_ORDER_FILE   =      "ordfile" ;


int main(int argc, char** argv){
     std::cout << "Usage:  THIS.EXE  XYZ_FILE " 
          << "[-" << FLAG_PARAM_FILE         << "=H_rad|H_ang|O_rad|O_ang]"
          << "[-" << FLAG_ATOM_ORDER_FILE    << "=NONE]"        
          << std::endl << std::endl;

     if (argc < 2) {
          return 0;    
     } 

     Gfunction_t<double> * gf = new Gfunction_t<double>();     // the G-function
   

     std::cout<<std::setprecision(18)<<std::scientific;



     //---------------------------------GET PARAMETERS ---------------------------------------//

     // parameter file
     std::string paramfile;   
     getCmdLineArgumentString(argc, (const char **)argv, FLAG_PARAM_FILE, paramfile);
     std::cout<<"Param File: "<<paramfile<<std::endl;
     
     // atom order file
     std::string ordfile;
     getCmdLineArgumentString(argc, (const char **)argv, FLAG_ATOM_ORDER_FILE, ordfile);    
     std::cout<<"Order File: "<<ordfile<<std::endl;

     std::cout<<"XYZ File: "<<argv[1]<<std::endl;

     //load scaling files
     const char * scaleFiles[2];
     scaleFiles[0] = SCALE_FILE_O;
     scaleFiles[1] = SCALE_FILE_H;


     //-------------------------------//CALCULATE G FUNCTION //-------------------------------//

     //make g function from xyz
     gf->make_G_XYZ(argv[1], paramfile.c_str(), ordfile.c_str(), scaleFiles);

     cout<<"Creating Network"<<endl;
     size_t numTypes = gf->NType;
     cout<<"NTypes: "<<numTypes<<endl;
     size_t numAtoms = gf->model.NATOM;
     cout<< "NAtoms: "<<numAtoms<<endl;
     size_t N = gf->NCluster;
     cout<< "NClusters: "<< N<<endl;


     //-------------------------------//CALCULATE ENERGIES //-------------------------------//
     allNets_t<double> * networkCollection = new allNets_t<double>(numTypes, INFILE1,CHECKCHAR); 
     double * finalOutput = new double[N];
     networkCollection -> runAllNets(gf,finalOutput);


     //-------------------------------//DISPLAY RESULTS//-------------------------------//
     cout<<"Final Results: "<<endl;
     for(int i = 0; i<N; i++){
          cout<<finalOutput[i] << " "<<endl;
     }


     if(finalOutput!=NULL) delete[] finalOutput;   

return 0;
}

