#include <iostream>
#include <sstream>
#include <fstream>
#include <stdlib.h>
#include <string>
#include <algorithm>   
#include <limits>
#include <vector>
#include <math.h>
#include <H5Cpp.h>
#include <memory>
#include <iomanip>


#include "utility.h"
#include "network.h"
#include "timestamps.h"
#include "NN_2L2H2O_poly2d.in"   
#include "readhdf5.hpp"
#define INFILE1     "32_2b_nn_single.hdf5"     // HDF5 files for different precisions
#define INFILE2     "32_2b_nn_double.hdf5"
#define CHECKCHAR1  "W"                 // dense_1_[W]           for "W"
#define CHECKCHAR2  "l"                 // dense_1/kerne[l]      for "l"
#define PATHTOMODEL "/model_weights"    // usual path to the group saving all the layers in HDF5 file
#define LAYERNAMES  "layer_names"       // Attribute name saving the list of layer names in HDF5
#define WEIGHTNAMES "weight_names"      // Attribute name saving the list of weight names in HDF5
#define LASTATVID   12				//The sequence ID of last activiation layer.
#define SAMPLECOUNT 11                  // input sample count
#define SAMPLEDIM   69                  // each input sample's dim 
#define MAXSHOWRESULT 20                // Max count of result to show

// Define the cblas library 
#ifdef _USE_GSL
#include <gsl/gsl_cblas.h>
#elif _USE_MKL
#include <mkl_cblas.h>
#else
//#include <gsl/gsl_cblas.h>
#endif

using namespace std;

#if defined (_USE_GSL) || defined (_USE_MKL)
template <>
void network_t<double>::fullyConnectedForward(const Layer_t<double> & layer,
                          	int& input, int&output, int&N,
                          	double* srcData, double** dstData)
	{
		
		output = layer.outputs;
		//store bias in dstData matrix(copy to each col) before computation
		//dstData is transposed to fully utilize rowMajor storage
		for(int i=0;i<N;i++)
			for(int j=0;j<output;j++)
				dstData[i][j] = layer.bias[j];
	

		double ** temp = NULL;
		size_t output_t = output, N_t = N;
		transpose_mtx<double>(dstData,temp, N_t, output_t);
		
		
		cblas_dgemm(CblasRowMajor,CblasTrans,CblasTrans,output,N,input,1,layer.weights[0],output,srcData,input,1,*temp,N);
		
		transpose_mtx<double>(temp,dstData,output_t,N_t);
		
		clearMemo<double>(temp);
		
		input = output;
     	
    	}; 

template<>
void network_t<float>::fullyConnectedForward(const Layer_t<float> & layer,
                          	int& input, int&output, int&N,
                          	float* srcData, float** dstData)
	{
		output = layer.outputs;

		//store bias in dstData matrix(copy to each col) before computation
		//dstData is transposed to fully utilize rowMajor storage
		for(int i=0;i<N;i++)
			for(int j=0;j<output;j++)
				dstData[i][j] = layer.bias[j];
	

		float ** temp = NULL;
		size_t output_t = output, N_t = N;
		transpose_mtx<float>(dstData,temp, N_t, output_t);
		
		
		cblas_sgemm(CblasRowMajor,CblasTrans,CblasTrans,output,N,input,1,layer.weights[0],output,srcData,input,1,*temp,N);
		
		transpose_mtx<float>(temp,dstData,output_t,N_t);
		
		clearMemo<float>(temp);

		input = output;
	};
		
#endif



/*TESTER*/
int main(int argc, char** argv){ 
	/*timers_t timers;
	timerid_t id;
	timers.insert_random_timer( id, 0, "Tester Execution");
     timers.timer_start(id);
	int input=4, output=3, N = 2;
	size_t input_t = input, output_t = output, N_t = N;

	//--------------------------------------------------------------------------------
	//------------------------------- DOUBLE TEST -------------------------------------

	cout <<"Doubles: " <<endl;
	double ** testWeights = NULL;
	
	if(!init_mtx_in_mem<double>(testWeights,input_t,output_t))
		cout << "Test Weights failed to initialize!"<<endl;
	
	for(int i=0;i<input;i++){
		for(int j=0;j<output;j++){
			testWeights[i][j] = (j+i)/3.14159265358979;
		}
	}
	cout<<"weights"<<endl;
	for(int i=0;i<input;i++){
		for(int j=0;j<output;j++){
			cout<<testWeights[i][j] << " " ;
		}
		cout<<endl;
	}
	cout<<"-------------------"<<endl;
	
	double * testBias = new double[3];
	
	testBias[0]= .314159; 
	testBias[1] =.271828; 
	testBias[2] =.173205;

	Layer_Net_t<double> * myNet = new Layer_Net_t<double>();

	string name = "test";

	
	myNet->insert_layer(name,input,output,testWeights,testBias);
	
	double ** testIn = NULL;
	if(!init_mtx_in_mem<double>(testIn,N_t,input_t))
		cout << "Test In failed to initialize!"<<endl;

	for(int i=0;i<N;i++){
		for(int j=0;j<input;j++){
			testIn[i][j] = (i*2+j)/(5*2.71828);
		}
	}
	
	cout<<"Input"<<endl;
	for(int i=0;i<N;i++){
		for(int j=0;j<input;j++){
			cout << testIn[i][j] << " " ;
		}
			cout<<endl;
	}
	cout<<"-------------------"<<endl;


	double ** testOut = NULL;
	if(!init_mtx_in_mem<double>(testOut,N_t,output_t))
		cout << "Test Out failed to initialize!"<<endl;
	

	myNet->neural_net.fullyConnectedForward(*(myNet->root),input,output,N,*testIn,testOut);
	
	for(int i=0;i<N;i++){
		for(int j=0;j<output;j++){
			cout<<testOut[i][j]<<" ";
		}
		cout<<endl;
	}

	string name2 = "testTanH";
	myNet->insert_layer(name2, 2);

	double ** testOut2 = NULL;
	if(!init_mtx_in_mem<double>(testOut2,N_t,output_t))
		cout << "Test Out2 failed to initialize!"<<endl;
	myNet->neural_net.activationForward_TANH(output,N,testOut,testOut2);
	

	cout<<"---------------------"<<endl;
	for(int i=0;i<N;i++){
		for(int j=0;j<output;j++){
			cout<<testOut2[i][j]<<" ";
		}
		cout<<endl;
	}

	//free memories
	delete myNet;
	delete [] testBias;
	clearMemo<double>(testWeights);
	clearMemo<double>(testIn);
	clearMemo<double>(testOut);
	clearMemo<double>(testOut2);

	//--------------------------------------------------------------------------------
	//------------------------------- FLOAT TEST -------------------------------------

	cout <<"Floats: " <<endl;
	float ** testWeights2 = NULL;
	
	if(!init_mtx_in_mem<float>(testWeights2,input_t,output_t))
		cout << "Test Weights2 failed to initialize!"<<endl;
	
	for(int i=0;i<input;i++){
		for(int j=0;j<output;j++){
			testWeights2[i][j] = (j+i)/3.14159265358979;
		}
	}
	
	float * testBias2 = new float[3];
	
	testBias2[0]= .314159; 
	testBias2[1] =.271828; 
	testBias2[2] =.173205;

	Layer_Net_t<float> * myNet2 = new Layer_Net_t<float>();

	string name3 = "test2";

	
	myNet2->insert_layer(name3,input,output,testWeights2,testBias2);
	
	float ** testIn2 = NULL;
	if(!init_mtx_in_mem<float>(testIn2,N_t,input_t))
		cout << "Test In 2 failed to initialize!"<<endl;

	for(int i=0;i<N;i++){
		for(int j=0;j<input;j++){
			testIn2[i][j] = (i*2+j)/(5*2.71828);
		}
	}


	float ** testOut3 = NULL;
	if(!init_mtx_in_mem<float>(testOut3,N_t,output_t))
		cout << "Test Out3 failed to initialize!"<<endl;
	

	myNet2->neural_net.fullyConnectedForward(*(myNet2->root),input,output,N,*testIn2,testOut3);
	
	for(int i=0;i<N;i++){
		for(int j=0;j<output;j++){
			cout<<testOut3[i][j]<<" ";
		}
		cout<<endl;
	}

	string name4 = "testTanH2";
	myNet2->insert_layer(name4, 2);

	float ** testOut4 = NULL;
	if(!init_mtx_in_mem<float>(testOut4,N_t,output_t))
		cout << "Test Out4 failed to initialize!"<<endl;
	myNet2->neural_net.activationForward_TANH(output,N,testOut3,testOut4);
	

	cout<<"---------------------"<<endl;
	for(int i=0;i<N;i++){
		for(int j=0;j<output;j++){
			cout<<testOut4[i][j]<<" ";
		}
		cout<<endl;
	}
	
	//free memories
	delete myNet2;
	delete [] testBias2;
	clearMemo<float>(testWeights2);
	clearMemo<float>(testIn2);
	clearMemo<float>(testOut3);
	clearMemo<float>(testOut4);

	timers.timer_end(id);
	timers.get_time_collections();
	return 0;
*/
try{
     cout << " Run tester with single floating point precision : " <<endl;
     runtester<float> (INFILE1, CHECKCHAR1, X[0]);
     cout << endl << endl;
     cout << " ================================================= " <<endl << endl;
     cout << " Run tester with double floating point precision : " <<endl;
     runtester<double>(INFILE2, CHECKCHAR2, Y[0]);
     } catch (...) {
          exit(1);     
     }
     exit(0);      
	return 0;
	
}

