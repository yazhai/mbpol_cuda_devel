# Implementing CPU based Neural Network model 

## General
These codes model the complete workflow of performing prediction based on Neural Network model in C++ and CBLAS libraries.

  - The code currently takes as input the xyz coordinates of atoms, and generates as output their energies.
  - The code first calculates distances from the xyz coordinates, then passes these distances through the g-function, and finally passes the g-function output through a 
	Neural Network which is modeled to predict the energies of the atoms
  - Currently the code does no training of the network and requires the weights/biases/structure of a pre-trained network as an input

## File list
Source Files:
- `utility.h/cpp`                  : File for utility functions
- `timestamps.h/cpp`               : Timer class for benchmarking
- `atomTypeID.h/cpp`               : Class manipulates atom name, type, internal id, etc.
- `readGparams.h/cpp`              : Class reads out the atom type correlations parameters.
- `Gfunction.h/cpp`                : Class constructs the G-function
- `network.h/cpp`				: Multiple Classes that contain the implementation of the Neural Network
- `readhdf5.hpp`				: Collection of functions to help read hdf5 files
- `xyzData.h/cpp`				: Functions that perform the xyz to distance calculation
- `fullTester.cpp`                 : The tester:  XYZ --> Gfn --> NN --> Energies


Resources:
- `3b_to_fit.xyz`                  : Tester information. XYZ coordinates for water trimers.
- `Gfunc_params_3Bv3.dat` 	    	: Atom type correlation parameter files
- `Gfn_order.dat`                  : The order of atom correlation parameters when constructing G-function

Output_Files:
- `NN_final.out` 				: Final energies for each atom
- `my_y_pred.txt` 				: final energies summed together -- the final output of the network to be comparedd with y_pred.txt
- `gfn_o_H1a`					: Intermediate gfunction output for atom H1A... same exists for all other atoms if intermediate output enabled
   
### For class file `Gfunction.h/cpp` constructing G-function:
These files contain following members:
   - It has a member `model` containing atom name, type, and id information for easily accessing atom information
   - It has a member `GP` containing the correlation parameters used for generating the G-function
   - It may employ the `openmp - pragma omp parallel for simd` to speed up the performance
   - Final results are saved in class member `G` 

In addition, it is possible to utilize `CBLAS` library and/or `fopenmp` to speed up performance
   
### For classes file `network.h/cpp` handling Neural Network 
These files contain the following important parts:
	- Layer Struct which has members: name, type, acttype, inputs/outputs(their dimensions), 
	  weights, bias, and prev/next pointers
	- network_t class which is responsible for movement through the network
	- Layer_Net_t class which is made up of a collection of Layer Structs having important method
	  predict which makes a prediction given input data and a finished network. 
	-runtester method which is called by the main fullTester and which calls the predict method

### For the provided tester:  
This tester first constructs the G-function using the resource files, and input, and then passes the output through the Neural Network to get the final energies
   - It accepts input from the `3b_to_fit.xyz` file, but should be able to accept input from any similarly formatted file of water molecule coordinates

## To Compile:
To compile the tester EXE:
   - `rm -r build; mkdir build; cd build`
   - `cmake [-D_FLAGS1=1 -D_FLAGS2=1] ..`
     - FLAGS may be ONE from EACH of the following groups:
       - Compiler type option, may be one of:
         - GNU [default];
         - INTEL;
       - CBLAS library utilization, may be one of:
         - MKL;[compatible with Intel2017];
         - GSL;
         - (None, by default);
       - OpenMP support, may be one of:
         - OPENMP;
         - (None, by default);
       - the OPTIMIZE OPTION may be one of the following:
         - XHOST (compatible with Intel compiler)
         - ~~CAVX512~~
         - ~~CRAVX512~~  
         - ~~CRAVX2~~  
         - ~~CRAVXI~~  
         - ~~AVX~~  
         - ~~NO_VECT~~
         - O (optimization level, O0/1/2/3)
     - Example:
       - `cmake -D_GSL=1 -D_OPENMP=1 -D_O=2 ..`
       - `cmake -D_INTEL=1 -D_MKL=1 -D_XHOST=1 ..`
   - `make` : compiles executable to _../bin_ folder (in testing)
   - `make cleanAll` : cleans up all outputfiles, object files, and executable (in testing)

## To RUN
To run with tester input:
   - Change directories to ./bin
   - `./Full_NN 3b_to_fit.xyz -paramfile=Gfunc_params_3Bv3.dat -ordfile=Gfn_order.dat -gfnOut=1`
   - -gfnOut=1 outputs intermediate gfunction outputs to files, otherwise only final energies are outputted to files
   - compare
   
  

