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

#define INFILE1     "/server-home1/ndanande/Documents/mbpol_cuda_devel/NN_2L2H2O_poly2d_CPU/34.hdf5"     // HDF5 files for Layer Data

//possible last chars of weight data(used to differentiate between weights and biases)
#define CHECKCHAR1  "W"                 // dense_1_[W]           for "W"
#define CHECKCHAR2  "l"                 // dense_1/kerne[l]      for "l"

//percentage of gfn output data used in training(and not in testing)
#define TRAIN_PERCENT 0

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
          << "[-" << FLAG_PARAM_FILE         << "=H_rad|H_ang|O_rad|O_ang]"
          << "[-" << FLAG_ATOM_ORDER_FILE    << "=NONE]"
		<< "[-" << FLAG_GFN_OUTPUT_ENABLE 	<< "=0]"		
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

     // normalize G by the maximum value of first 90% dimers
     //gf.norm_G_by_maxabs_in_first_percent(TRAIN_PERCENT);    
     // results saved in gf.G which is a map<string:atom_idx, double**>

	//define number of samples
	size_t N = gf.NCluster;
	//int testIndexStart = N*TRAIN_PERCENT;
	//N -= testIndexStart;		//effective new N(for test size) after 90% used for training

	cout<<"Something"<<endl;
	//define number of atoms
	size_t numAtoms = gf.G.size();


	//init memory to store test set gfn Outputs
	double ** X = new double * [numAtoms];

	//store inputDimension for each seperate atom
	size_t * inputDim = new size_t[numAtoms];

		
	
	int i=0;	//iterating variable

	cout<<"Begin, numAtoms: " << numAtoms<<" Num Clusters: " << N<<endl;
	//put G function outputs into 2d array for usage in NN(only the test portion)
 	for(auto it=gf.G.begin(); it!=gf.G.end(); it++){
		inputDim[i] = gf.G_param_max_size[gf.model.TYPE_EACHATOM[i]];
		cout<<"Input Dim " << i << " : " << inputDim[i]<<endl; 
		X[i] = new double[inputDim[i]*N];
		for(int ii=0; ii<N;ii++){
			for(int jj=0;jj<inputDim[i];jj++){
				X[i][ii*inputDim[i] + jj] = (*it)[ii][jj];
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
			string atomType = std::to_string(gf.model.TYPE_EACHATOM[i]);
			string atomNum = std::to_string(i);
			gfuncOutPath = gfuncOutPrefix+atomType+"_"+atomNum;
			gfuncOut.open(gfuncOutPath);
			cout<<"inputDim " << atomType<<atomNum<<" inputDim[i] " << "N: "<< N<<endl;
			for(int j=0;j<N;j++){
				for(int k=0;k<inputDim[i];k++){
					gfuncOut<<setprecision(18)<<scientific<<(*it)[j][k]<<" ";
				}
				gfuncOut<<endl;
			}
			i++;
			gfuncOut.close();
		}
	}

     cout<<("TEST 5")<<endl; 
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
