#include <iostream>
#include <sstream>
#include <fstream>
#include <stdlib.h>
#include <string>   
#include <limits>
#include <vector>
#include <math.h>

#include "utility.h"
#include "network.h"
#include "timestamps.h"

//old implementation
//#define INFILE_S 	"/server-home1/ndanande/Documents/mbpol_cuda_devel/NN_2L2H2O_poly2d_CPU/NN_in_Single.in"	//Input Data, Single
//#define INFILE_D	"/server-home1/ndanande/Documents/mbpol_cuda_devel/NN_2L2H2O_poly2d_CPU/NN_in_Double.in"	//Input Data, Double
//#define INFILE1     "/server-home1/ndanande/Documents/mbpol_cuda_devel/NN_2L2H2O_poly2d_CPU/32_2b_nn_single.hdf5"     // HDF5 files for different precisions Layer Data
//#define INFILE2     "/server-home1/ndanande/Documents/mbpol_cuda_devel/NN_2L2H2O_poly2d_CPU/32_2b_nn_double.hdf5"
//#define CHECKCHAR1  "W"                 // dense_1_[W]           for "W"
//#define CHECKCHAR2  "l"                 // dense_1/kerne[l]      for "l"

//input file parameters
#define SAMPLECOUNT 11                  // input sample count(N)
#define SAMPLEDIM   69                  // each input sample's dim(input) 

// Define the cblas library 
#ifdef _USE_GSL
#include <gsl/gsl_cblas.h>
#elif _USE_MKL
#include <mkl_cblas.h>
#endif


//define openMP
#ifdef _OPENMP
#include <omp.h>
#endif 


using namespace std;


//cblas implementations of double and single precision forward dense layer
//Input and layer.Weights already transposed for optimal row-major computation.
#if defined (_USE_GSL) || defined (_USE_MKL)
template <>
void network_t<double>::fullyConnectedForward(const Layer_t<double> & layer,
                          	size_t & input, size_t & output, size_t & N,
                          	double* & srcData, double** & dstData)
	{
		
		output = layer.outputs;	//update the dimensions of the layer

		//create space for output
		if(dstData != nullptr)
			clearMemo<double>(dstData);	
		init_mtx_in_mem<double>(dstData,output,N);

		//store bias in dstData matrix(copy to each col) before computation
		//dstData is transposed to fully utilize rowMajor storage

		for(int i=0;i<output;i++)
			#ifdef _OPENMP
          	#pragma omp parallel for simd shared(dstData,layer)
          	#endif 
			for(int j=0;j<N;j++)
				dstData[i][j] = layer.bias[i];
		
		//compute Weights[output][input] x Input[input][N] + Bias[output][N]
		cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,output,N,input,1.0,layer.weights[0],input,srcData,N,1,*dstData,N);
		
		//update dimensions of input for next layer's use
		input = output;
    	}; 

template<>
void network_t<float>::fullyConnectedForward(const Layer_t<float> & layer,
                          	size_t & input, size_t & output, size_t & N,
                          	float* & srcData, float** & dstData)
	{

		output = layer.outputs;	//update the dimensions of the layer
		
		//create space for output
		if(dstData != nullptr)
			clearMemo<float>(dstData);	
		init_mtx_in_mem<float>(dstData,output,N);

		//store bias in dstData matrix(copy to each col) before computation
		//dstData is transposed to fully utilize rowMajor storage
		for(int i=0;i<output;i++)
			#ifdef _OPENMP
          	#pragma omp parallel for simd shared(dstData,layer)
          	#endif 
			for(int j=0;j<N;j++)
				dstData[i][j] = layer.bias[i];

		//compute Weights[output][input] x Input[input][N] + Bias[output][N]
		cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,output,N,input,1.0,layer.weights[0],input,srcData,N,1,*dstData,N);
		
		//update dimensions of input for next layer's use		
		input = output;
	};
		
#endif


/*TESTER*/
/*
int main(int argc, char** argv){ 

	//oldTester();				//TO USE OLD TESETERS: make neural_net public in network.h file
	//oldTester2();			//testers may be out of date

	//Timers for benchmarking
	timers_t timers;
	timerid_t id,id2;
	timers.insert_random_timer( id, 0, "NN_singlePrecisionTime");
	timers.insert_random_timer(id2,1,"NN_doublePrecisionTime");

	float ** X = nullptr;
	double ** Y = nullptr;
	size_t sampleCount_Single = 1;
	size_t sampleCount_Double = 1;
	size_t sampleDim_Single =1;
	size_t sampleDim_Double = 1;
	read2DArray_with_max_thredhold<float>(X, sampleCount_Single, sampleDim_Single, INFILE_S, 1);
	read2DArray_with_max_thredhold<double>(Y,sampleCount_Double,sampleDim_Double,INFILE_D,1);
try{
     cout << " Run tester with single floating point precision : " <<endl;
	timers.timer_start(id);
     runtester<float> (INFILE1, CHECKCHAR1, X[0],sampleCount_Single,sampleDim_Single);
	timers.timer_end(id);
     cout << endl << endl;
     cout << " ================================================= " <<endl << endl;
     cout << " Run tester with double floating point precision : " <<endl;
	timers.timer_start(id2);
     runtester<double>(INFILE2, CHECKCHAR2, Y[0],sampleCount_Double,sampleDim_Double);
	timers.timer_end(id2);
     } catch (...) {
          exit(1);     
     }
	
	//get Times
	timers.get_all_timers_info();
     timers.get_time_collections();     
     exit(0); 
	return 0;
	
}

*/


/*
void oldTester(){

	int input=4, output=3, N = 2;
	size_t input_t = input, output_t = output, N_t = N;

	//--------------------------------------------------------------------------------
	//------------------------------- DOUBLE TEST -------------------------------------
	cout<<"OUTPUT DIM: ";
	cout<<output_t<<endl;
	cout<<"INPUT DIM: ";
	cout<<input_t<<endl;

	cout <<"Doubles: " <<endl;
	double ** testWeightsT = NULL;
	
	if(!init_mtx_in_mem<double>(testWeightsT,input_t,output_t))
		cout << "Test Weights failed to initialize!"<<endl;
	
	for(int i=0;i<input;i++){
		for(int j=0;j<output;j++){
			testWeightsT[i][j] = (j+i)/3.14159265358979;
		}
	}
	cout<<"weights"<<endl;
	for(int i=0;i<input;i++){
		for(int j=0;j<output;j++){
			cout<<testWeightsT[i][j] << " " ;
		}
		cout<<endl;
	}
	cout<<"-------------------"<<endl;

	double * testWeights = testWeightsT[0];
	
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

	double ** testInT = nullptr;

	transpose_mtx<double>(testInT,testIn, N_t, input_t);


	double ** testOutT = nullptr;
	double ** testOut = nullptr;
	
	myNet->neural_net.fullyConnectedForward(*(myNet->root),input_t,output_t,N_t,*testInT,testOutT);
	transpose_mtx<double>(testOut,testOutT,output_t,N_t);	

	for(int i=0;i<N;i++){
		for(int j=0;j<output;j++){
			cout<<testOut[i][j]<<" ";
		}
		cout<<endl;
	}

	cout<<"OUTPUT DIM: ";
	cout<<output_t<<endl;
	cout<<"INPUT DIM: ";
	cout<<input_t<<endl;
	string name2 = "testTanH";
	myNet->insert_layer(name2, 2);

	double ** testOut2T = nullptr;
	double ** testOut2 = nullptr;
		
	myNet->neural_net.activationForward_TANH(output_t,N_t,testOutT,testOut2T);
	transpose_mtx<double>(testOut2,testOut2T,output_t,N_t);	
	

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
	clearMemo<double>(testWeightsT);
	clearMemo<double>(testIn);
	//clearMemo<double>(testOut);
	//clearMemo<double>(testOut2);

	cout<<"OUTPUT DIM: ";
	cout<<output_t<<endl;
	cout<<"INPUT DIM: ";
	cout<<input_t<<endl;

	//--------------------------------------------------------------------------------
	//------------------------------- FLOAT TEST -------------------------------------
	cout <<"Floats: " <<endl;
	float ** testWeights2T = NULL;
	input_t = input;
	output_t = output;
	if(!init_mtx_in_mem<float>(testWeights2T,input_t,output_t))
		cout << "Test Weights2 failed to initialize!"<<endl;
	
	for(int i=0;i<input;i++){
		for(int j=0;j<output;j++){
			testWeights2T[i][j] = (j+i)/3.14159265358979;
		}
	}
	
	
	float * testWeights2 = testWeights2T[0];
	
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
	

	myNet2->neural_net.fullyConnectedForward(*(myNet2->root),input_t,output_t,N_t,*testIn2,testOut3);
	
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
	clearMemo<float>(testWeights2T);
	clearMemo<float>(testIn2);
	clearMemo<float>(testOut3);
	clearMemo<float>(testOut4);

}

void oldTester2(){
	
//--------------------------------------------------------------------------------
//------------------------------- Double - Double TEST -------------------------------------

	int input=4, output=3, N = 2;
	size_t input_t = input, output_t = output, N_t = N;

	cout<<"OUTPUT DIM: ";
	cout<<output_t<<endl;
	cout<<"INPUT DIM: ";
	cout<<input_t<<endl;

	cout <<"Doubles: " <<endl;
	double ** testWeightsT = NULL;
	
	if(!init_mtx_in_mem<double>(testWeightsT,input_t,output_t))
		cout << "Test Weights failed to initialize!"<<endl;
	
	for(int i=0;i<input;i++){
		for(int j=0;j<output;j++){
			testWeightsT[i][j] = (j+i)/3.14159265358979;
		}
	}
	cout<<"weights"<<endl;
	for(int i=0;i<input;i++){
		for(int j=0;j<output;j++){
			cout<<testWeightsT[i][j] << " " ;
		}
		cout<<endl;
	}
	cout<<"-------------------"<<endl;

	double * testWeights = testWeightsT[0];
	
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
	
	
	myNet->neural_net.fullyConnectedForward(*(myNet->root),input_t,output_t,N_t,*testIn,testOut);
	
	for(int i=0;i<N;i++){
		for(int j=0;j<output;j++){
			cout<<testOut[i][j]<<" ";
		}
		cout<<endl;
	}


double ** testWeights2T = NULL;

	input = 3; output = 4;	

	if(!init_mtx_in_mem<double>(testWeights2T,input_t,output_t))
		cout << "Test Weights failed to initialize!"<<endl;
	
	for(int i=0;i<input;i++){
		for(int j=0;j<output;j++){
			testWeights2T[i][j] = (j+i)/3.14159265358979;
		}
	}
	cout<<"weights"<<endl;
	for(int i=0;i<input;i++){
		for(int j=0;j<output;j++){
			cout<<testWeights2T[i][j] << " " ;
		}
		cout<<endl;
	}
	cout<<"-------------------"<<endl;

	double * testWeights2 = testWeights2T[0];
	
	double * testBias2 = new double[4];
	
	testBias2[0]= .314159; 
	testBias2[1] =.271828; 
	testBias2[2] =.173205;
	testBias2[3] = .1;

	Layer_Net_t<double> * myNet2 = new Layer_Net_t<double>();

	string name2 = "test2";

	
	myNet2->insert_layer(name2,input_t,output_t,testWeights2,testBias2);

double ** testOut2 = NULL;
	if(!init_mtx_in_mem<double>(testOut2,N_t,output_t))
		cout << "Test Out failed to initialize!"<<endl;
	
	
	myNet2->neural_net.fullyConnectedForward(*(myNet2->root),input_t,output_t,N_t,*testOut,testOut2);
	
	for(int i=0;i<N;i++){
		for(int j=0;j<output;j++){
			cout<<testOut2[i][j]<<" ";
		}
		cout<<endl;
	}
	
	
	//free memories
	delete myNet2;
	delete [] testBias2;
	clearMemo<double>(testWeights2T);
	clearMemo<double>(testOut);
	clearMemo<double>(testOut2);

}


*/



