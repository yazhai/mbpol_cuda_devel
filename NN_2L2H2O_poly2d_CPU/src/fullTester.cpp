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

#include "readGparams.h"
#include "atomTypeID.h"
#include "utility.h"
#include "timestamps.h"
#include "Gfunction.h"
#include "network.h"

#define INFILE1     "/server-home1/ndanande/Documents/mbpol_cuda_devel/NN_2L2H2O_poly2d_CPU/16.hdf5"     // HDF5 files for Layer Data

//possible last chars of weight data(used to differentiate between weights and biases)
#define CHECKCHAR1  "W"                 // dense_1_[W]           for "W"
#define CHECKCHAR2  "l"                 // dense_1/kerne[l]      for "l"

//percentage of gfn output data used in training(and not in testing)
#define TRAIN_PERCENT 0.9

//define openMP
#ifdef _OPENMP
#include <omp.h>
#endif 


using namespace std;

const char* FLAG_COLUMN_INDEX_FILE =   "columnfile" ;
const char* FLAG_PARAM_FILE        =    "paramfile" ;
const char* FLAG_ATOM_ORDER_FILE   =      "ordfile" ;
const char* FLAG_GFN_OUTPUT_ENABLE = 	    "gfnOut";	//enable intermediate gfn output to file

const int THREDHOLD_COL = -1;
const double THREDHOLD_MAX_VALUE = 60.0;

int main(int argc, char** argv){ 

     cout << "Usage:  THIS.EXE  XYZ_FILE " 
          << "[-" << FLAG_COLUMN_INDEX_FILE  << "=NONE]"  
          << "[-" << FLAG_PARAM_FILE         << "=H_rad|H_ang|O_rad|O_ang]"
          << "[-" << FLAG_ATOM_ORDER_FILE    << "=NONE]"
		<< "[-" << FLAG_GFN_OUTPUT_ENABLE 	<< "=0]"		
          << endl << endl;

     if (argc < 2) {
          return 0;    
     } 

     Gfunction_t<double> gf;     // the G-function
   
     // column index file
     string colidxfile;
     getCmdLineArgumentString(argc, (const char **)argv, FLAG_COLUMN_INDEX_FILE, colidxfile);
          
     // parameter file
     string paramfile;
     getCmdLineArgumentString(argc, (const char **)argv, FLAG_PARAM_FILE, paramfile);
     
     // atom order file
     string ordfile;
     getCmdLineArgumentString(argc, (const char **)argv, FLAG_ATOM_ORDER_FILE, ordfile);    

	//Gfn output enable
	int gfnOut = getCmdLineArgumentInt(argc, (const char **)argv, FLAG_GFN_OUTPUT_ENABLE);     
       
	//make G function with xyz file
     gf.make_G_XYZ(argv[1],colidxfile.c_str(), paramfile.c_str(), ordfile.c_str());

     // normalize G by the maximum value of first 90% dimers
     gf.norm_G_by_maxabs_in_first_percent(TRAIN_PERCENT);    
     // results saved in gf.G which is a map<string:atom_idx, double**>
     
	//define number of samples
	size_t N = gf.ndimers;
	int testIndexStart = N*TRAIN_PERCENT;
	N -= testIndexStart;		//effective new N(for test size) after 90% used for training
	
	//define number of atoms
	size_t numAtoms = gf.G.size();

	//init memory to store test set gfn Outputs
	double ** X = new double * [numAtoms];
	
	//store inputDimension for each seperate atom
	size_t * inputDim = new size_t[numAtoms];

		
	int i=0;	//iterating variable
	//put G function outputs into 2d array for usage in NN(only the test portion)
 	for(auto it=gf.G.begin(); it!=gf.G.end(); it++){
		//string atom = gf.model.atoms[it->first]->name;
		string atom_type = gf.model.atoms[it->first]->type;
		inputDim[i] = gf.G_param_max_size[atom_type];
		X[i] = new double[inputDim[i]*N];
		for(int ii=0; ii<N;ii++){
			for(int jj=0;jj<inputDim[i];jj++){
				X[i][ii*inputDim[i] + jj] = it->second[jj][ii+testIndexStart];
			}
		}
		i++;
	}
	
	//if user specified to output intermediate Gfn outputs to files, do so
	if(gfnOut != 0){
		i=0;		//reset iterating variable
		ofstream gfuncOut;
		string gfuncOutPrefix = "gfn_o_";
		string gfuncOutPath = "";
		for(auto it=gf.G.begin();it!=gf.G.end();it++){
			string atom = gf.model.atoms[it->first]->name;
			string atom_type = gf.model.atoms[it->first]->type;
			atom.erase(remove(atom.begin(), atom.end(), ')'),atom.end());	//get rid of () for better file name format
			atom.erase(remove(atom.begin(), atom.end(), '('),atom.end());
			gfuncOutPath = gfuncOutPrefix+atom;
			gfuncOut.open(gfuncOutPath);
			for(int j=0;j<gf.ndimers;j++){
				for(int k=0;k<inputDim[i];k++){
					gfuncOut<<it->second[k][j]<<" ";
				}
				gfuncOut<<endl;
			}
			i++;
			gfuncOut.close();
		}
	}

	cout<<"Running Network"<<endl;
	runtester<double>(INFILE1,CHECKCHAR2,X,numAtoms,N,inputDim);


	/*free memory*/
	for(int j=0;j<numAtoms;j++){
		delete [] X[j];
	}
	delete [] X;
	delete [] inputDim;
				
				
				
return 0;
}
