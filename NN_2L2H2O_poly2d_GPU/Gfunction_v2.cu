
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
#include "utility.h"


using namespace std;

const char* FLAG_COLUMN_INDEX_FILE =   "columnfile" ;
const char* FLAG_PARAM_FILE        =    "paramfile" ;
const char* FLAG_ATOM_ORDER_FILE   =      "ordfile" ;
const char* FLAG_GFN_OUTPUT_ENABLE =          "gfnOut";     //enable intermediate gfn output to file

const int THREDHOLD_COL = -1;
const double THREDHOLD_MAX_VALUE = 60.0;

using namespace std;
int main(int argc, char** argv){
     cout << "Usage:  THIS.EXE  XYZ_FILE " 
          << "[-" << FLAG_PARAM_FILE         << "=H_rad|H_ang|O_rad|O_ang]"
          << "[-" << FLAG_ATOM_ORDER_FILE    << "=NONE]"
          << "[-" << FLAG_GFN_OUTPUT_ENABLE      << "=0]"          
          << endl << endl;

     if (argc < 2) {
          return 0;    
     } 

     Gfunction_t<double> gf;     // the G-function
   
          
     // parameter file
     string paramfile;
     getCmdLineArgumentString(argc, (const char **)argv, FLAG_PARAM_FILE, paramfile);
     
     // atom order file
     string ordfile;
     getCmdLineArgumentString(argc, (const char **)argv, FLAG_ATOM_ORDER_FILE, ordfile);    

     //Gfn output enable
     int gfnOut = getCmdLineArgumentInt(argc, (const char **)argv, FLAG_GFN_OUTPUT_ENABLE);     

     //make G function with xyz file
     gf.make_G_XYZ(argv[1], paramfile.c_str(), ordfile.c_str());

return 0;
}

