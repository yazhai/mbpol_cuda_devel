#ifndef NETWORK_H
#define NETWORK_H


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
#include "readhdf5.hpp"

//file things
#define PATHTOMODEL "/model_weights"    // usual path to the group saving all the layers in HDF5 file
#define LAYERNAMES  "layer_names"       // Attribute name saving the list of layer names in HDF5
#define WEIGHTNAMES "weight_names"      // Attribute name saving the list of weight names in HDF5

//NN constants
#define LASTATVID   12				//The sequence ID of last activiation layer.
#define SAMPLECOUNT 11                  // input sample count(N)
#define SAMPLEDIM   69                  // each input sample's dim(input) 
#define MAXSHOWRESULT 20                // Max count of result to show

// Define the cblas library 
#ifdef _USE_GSL
#include <gsl/gsl_cblas.h>
#elif _USE_MKL
#include <mkl_cblas.h>
#else
//#include <gsl/gsl_cblas.h>
#endif

//define openMP
#ifdef _OPENMP
#include <omp.h>
#endif 



//***********************************************************************************
// Structure of Network:
//		dimensions: N = number of samples, input = sample input Dimension, 
//			output = sample output dimension
//		Layer Weights: Inputed as  input x output dimensional array
//		Layer Bias:	Inputed as 1xoutput dimensional array
//						extended to Nxoutput dimensional array for computation
//		Layer Input:	Inputed as N x input dimensional array
//		
//		Output = Input*Weights + Bias
//		
//		OR: Output = transpose(Weights) * transpose(Input) + transpose(Bias)
//		This second method is used in the code in order to take advantage of 
//		rowMajor memory storage.
//***********************************************************************************

using namespace std;


//Type of Layer
enum class Type_t {
     UNINITIALIZED  = 0 ,
     DENSE          = 1 ,
     ACTIVIATION    = 2 , 
     MAX_TYPE_VALUE = 3
};

// Type of activiation layer
enum class ActType_t {
     NOACTIVIATION  = 0 ,
     LINEAR         = 1 ,
     TANH           = 2 ,
     MAX_ACTTYPE_VALUE = 3
};


template<typename T>
struct Layer_t{
	string name;
	Type_t type;
	ActType_t acttype;
 	
	size_t inputs;     // number of input dimension
    	size_t outputs;    // number of output dimension
    	T ** weights; // weight matrix(stored as outputs x inputs)
    	T * bias; // bias vector(1xoutput) or (outputx1)

	Layer_t * prev = nullptr;
	Layer_t * next = nullptr;
	
	Layer_t() :  name("Default_Layer"), weights(NULL), bias(NULL), inputs(0), 
		outputs(0), type(Type_t::UNINITIALIZED), acttype(ActType_t::NOACTIVIATION) {};


	Layer_t( string _name, size_t _inputs, size_t _outputs, 
          	T* _weights, T* _bias)
                  : inputs(_inputs), outputs(_outputs), type(Type_t::DENSE), 
			acttype(ActType_t::NOACTIVIATION){
     
        	name = _name ;

		/*init Weight Matrix = Output x Input Dimensions (Transposed)*/
		T ** temp;
		weights = nullptr;
        	if(!init_mtx_in_mem<T>(temp,inputs,outputs)){
			cout<<"FAILED TO INITIALIZE MEMORY FOR WEIGHTS"<<endl;
		}
		copy(_weights,_weights+outputs*inputs,temp[0]);
		transpose_mtx<T>(weights, temp, inputs, outputs);		
		clearMemo<T>(temp);

		/*init Bias Matrix = 1xOutput dimensions -- "Pseudo-Transposed" */
		bias = new T[outputs];
		copy(_bias,_bias+outputs,bias);
	}

	// construct an activation layer, by integer 
	Layer_t(string _name, int _acttype)
                  : weights(NULL), bias(NULL),inputs(0), outputs(0), 
			type(Type_t::ACTIVIATION){
     
     	if (_acttype < int(ActType_t::MAX_ACTTYPE_VALUE) ) {
               acttype = static_cast<ActType_t>(_acttype);
          }
          name = _name;
    	}

	//construct an activation layer by name
	Layer_t(string _name, ActType_t _acttype)
                  : weights(NULL),bias(NULL),inputs(0),outputs(0), 
                type(Type_t::ACTIVIATION){
      
          acttype = _acttype;
          name = _name;
    	} 

	~Layer_t(){
		if(weights != NULL){
			clearMemo<T>(weights);
		}
		if(bias != NULL) delete [] bias;
	}

};


//Class to Move propogate through network
template<typename T>
class network_t{

public:
	
	//non-cblas implementation of fully Connected Forward Propogation
	void fullyConnectedForward(const Layer_t<T> & layer,
                          	size_t & input, size_t & output, size_t & N,
                          	T* & srcData, T** & dstData){

		//update dimensionals for layer
		output = layer.outputs;

		//create space for output
		if(dstData != nullptr){
			clearMemo<T>(dstData);	
		}
		init_mtx_in_mem<T>(dstData,output,N);
		
		//Conduct Matrix Multiplication
		#ifdef _OPENMP
          #pragma omp parallel for simd collapse(3) shared(dstData,srcData,layer)
          #endif 
		for(int i=0;i<input;i++){
			for(int j=0;j<output;j++){
				for (int k=0;k<N;k++){
					dstData[j][k] += layer.weights[j][i]*srcData[N*i+k];
				}
			}
		}
		
		//Conduct Bias Addition
		#ifdef _OPENMP
          #pragma omp parallel for simd collapse(2) shared(dstData,layer)
          #endif 
		for(int i=0;i<output;i++){
			for(int j=0;j<N;j++){
				dstData[i][j] += layer.bias[i];
			}
		}

		//update dimensions of input for next layer's use
		input = output;
    	} 

	//non-cblas implementaiton of forward propogation activation function(TANH)
	void activationForward_TANH(const int & output,const int & N , T** srcData, T** & dstData){	
		
		//create space for output
		if(dstData != nullptr){
			clearMemo<T>(dstData);	
		}
		init_mtx_in_mem<T>(dstData,output,N);
		
		//complete faster TANH computation
		T x;
		for(int i=0;i<output;i++){
			for(int j=0;j<N;j++){
				x = srcData[i][j];
				x = exp(2*x);
				x = (x-1)/(x+1);
				dstData[i][j] = x;
			}	
		}
	}
		
};


// Using cblas_dgemm if libraries are employed
#if defined (_USE_GSL) || defined (_USE_MKL)
template <>
void network_t<double>::fullyConnectedForward(const Layer_t<double> & layer,size_t & input, size_t & output, size_t & N,double* & srcData, double** & dstData);

template<>
void network_t<float>::fullyConnectedForward(const Layer_t<float> & layer,size_t & input, size_t & output, size_t & N,float* & srcData, float** & dstData);
#endif


//class for Network of Layers
template<typename T>
class Layer_Net_t{
private: 
	
	//controls propogation through network
	network_t<T> neural_net;
	
	//helper function: switch two pointers to pointers
	void switchptr(T** & alpha, T** & bravo){
    		T** tmp;
          tmp = alpha;
          alpha = bravo;
          bravo = tmp;
          tmp = nullptr;
     }
	
public:
	
	//network_t<T> neural_net;	//make public for testing purposes
	
	Layer_t<T> * root = nullptr;
	
	Layer_Net_t(){};
	
	~Layer_Net_t(){
		Layer_t<T>* curr = nullptr;
		if(root != NULL){
			curr = root;
			while(curr->next){curr = curr->next;};
			while(curr->prev){
				curr = curr->prev;
				delete curr->next;
				curr->next = nullptr;
			}
			curr = nullptr;
			delete root;
			root = nullptr;
		}
	}
	
	//inserting a dense layer
	void insert_layer(string &_name, size_t _inputs, size_t _outputs, 
          T * & _weights, T * & _bias){
		if(root!=NULL){
			Layer_t<T> * curr = root;
			while(curr->next){curr = curr->next;};
			curr->next = new Layer_t<T>(_name,_inputs,_outputs,_weights,_bias);
			curr->next->prev = curr;
		}
		else{
			root = new Layer_t<T>(_name, _inputs, _outputs, _weights, _bias);
		}
	}

	// Inserting an activiation layer by type (int)
     void insert_layer(string &_name, int _acttype){
          if (root!=NULL) {
               Layer_t<T>* curr = root;
               while(curr->next) {curr = curr->next;};
               curr->next = new Layer_t<T>(_name, _acttype);
               curr->next->prev = curr;
          }
		else{
			root = new Layer_t<T>(_name,_acttype);
		}
	}

	//Inserting an activation layer by type(enum)
	void insert_layer(string &_name, ActType_t _acttype){
		cout<<"TRACE"<<endl;
		if (root!=NULL) {
               Layer_t<T> * curr = root;
               while(curr->next) {curr = curr->next;};
               curr->next = new Layer_t<T>(_name, _acttype);
               curr->next->prev = curr;
          } 
		else {
               root = new Layer_t<T>(_name, _acttype);
          }
     
     }

	// Get layer ptr according to its index (start from 1 as 1st layer, 2 as second layer ...)
     Layer_t<T>* get_layer_by_seq(int _n){
          Layer_t<T>* curr=root;
          int i = 1;
          
          while( (curr->next != NULL)  && (i<_n) ){
               curr = curr->next;
               i++ ;
          };
          return curr;
     } 
	
	 // Make prediction according to all the layers in the model
     void predict(T* _inputData, int _N, int _input, T* & _outputData, unsigned long int& _outsize){
		if (root != NULL) {
             
			size_t input = _input;
			size_t N = _N;
             	size_t output = 1;

             	//two pointers used to store and recieve data(switch between them)
             	T** srcDataPtr = nullptr; 
			T** dstDataPtr = nullptr;

			//init srcDataPtr to point to tranpose of input data
			T** temp;
			init_mtx_in_mem<T>(temp,N,input);
			copy(_inputData,_inputData+input*N,temp[0]);
			transpose_mtx<T>(srcDataPtr, temp, N, input);
			clearMemo<T>(temp);
                                              
             	Layer_t<T>* curr = root;

            	do{
      			//DENSE LAYER PROPOGATION
               	if ( curr-> type == Type_t::DENSE ) { 
                    	// If it is a dense layer, we perform fully_connected forward 
                   		neural_net.fullyConnectedForward((*curr), input,output,N, *srcDataPtr, dstDataPtr);
					//note:the array dimensions are switched inside of fullyConnectedForward function
                    	switchptr(srcDataPtr, dstDataPtr);

                      //ACTIVATION LAYER PROPOGATION
              		} 
				else if (curr -> type == Type_t::ACTIVIATION){
                    	// If it is an activiation layer, perform corresponding activiation forwards
                    	if (curr -> acttype == ActType_t::TANH){
                         	neural_net.activationForward_TANH(output,N, srcDataPtr, dstDataPtr);
                         	switchptr(srcDataPtr, dstDataPtr);
                    	} 
					else if (curr->acttype == ActType_t::LINEAR) {    
                         	cout << "Linear Activation Layer Called" <<endl;
                    	} 
					else {
						cout <<"Unknown Activation Type!"<<endl;
					}
              	 	} 
				else {
                    	cout << "Unknown layer type!" <<endl;
               	}

        		} 
			while(  (curr=curr->next) != NULL);
             
			//final output size calculated
             	_outsize=input*N;

			//create space for output Data
             	if(_outputData!=NULL){
                    delete[] _outputData;
             	}
             	_outputData = new T[_outsize];
			
             	//copy from srcDataPtr to outputData          
            	copy(*srcDataPtr,*srcDataPtr + _outsize,_outputData);
                       
             	// Don't forget to release resource !!!
			clearMemo<T>(srcDataPtr);
			clearMemo<T>(dstDataPtr);
			srcDataPtr = nullptr;
             	dstDataPtr = nullptr;  
        }

        return;
	}	    
};	


// tester function, including reading HDF5 file, creating layers, and making the prediction.
template <typename T>
void runtester(const char* filename, const char* checkchar, T* input){

	using namespace H5;
	
     // initialize memory for rank, dims, and data
     // !!! Don't forget to free memory before exit!!!
     hsize_t data_rank=0;
     hsize_t* data_dims = nullptr;
     T* data = nullptr;     
     
     hsize_t bias_rank=0;
     hsize_t* bias_dims = nullptr;
     T* bias = nullptr;
     
     Layer_Net_t<T> layers;
     
     // reserver for results
     unsigned long int outsize = 0; 
     T* output = nullptr;     
     
     // Open HDF5 file handle, read only
     H5File file(filename,H5F_ACC_RDONLY);
     
     
     try{     
          // Get saved layer names
          vector<string> layernames;
          layernames = Read_Attr_Data_By_Seq(file,PATHTOMODEL, LAYERNAMES); 

          for (auto it=layernames.begin();it!=layernames.end();it++) {
               // for one single layer get path
               string layerpath = mkpath ( string(PATHTOMODEL),  *it ) ;
               
               // get this layer's dataset names(weights and bias)
               vector<string> weights;
               weights = Read_Attr_Data_By_Seq(file,layerpath.c_str(), WEIGHTNAMES);
               
               
               cout << " Reading out layer data: " << *it << endl;
               for (auto it2 = weights.begin(); it2 != weights.end(); it2++){ 
                    // for one data set get path
                    string datasetPath = mkpath(layerpath,*it2) ;
                    
                    // check the dataset name's last character to see if this dataset is a Weight or a Bias
                    if ((*it2).compare(((*it2).length()-1),1, checkchar )==0){
                         // get out weight data
                         Read_Layer_Data_By_DatName<T> (file, datasetPath.c_str(), data, data_rank, data_dims); 
                    }
				else{
                         // get out bias data
                         Read_Layer_Data_By_DatName<T> (file, datasetPath.c_str(), bias, bias_rank, bias_dims);             
                    }
               }
               // When reading out a dense layer, a 2d weight matrix is obtained
               // Otherwise, it is a 0d matrix (null)
               if (data_rank==2){
                    cout << " Initialize dense layer : " << *it << endl;
				
				//insert dense layer
				layers.insert_layer(*it, data_dims[0], data_dims[1], data, bias);							
                    
				//reset values for next loop
				data_rank=0;
                    bias_rank=0;
               } 
			else {
                    cout << " Initialize activiation layer : " << *it << endl;
                    layers.insert_layer(*it, ActType_t::TANH);               
               }

               cout << " Layer " << *it << " is initialized. " <<endl <<endl;
          }
          
          cout << "Inserting Layers finished !" <<endl;
          
          // In our test model, we insert 5 (fully_connect + tanh_activiation) layers
          // plus 1 (fully_connect + linear_activiation) layers
          // So change the last activiation layer's type to linear
          layers.get_layer_by_seq(LASTATVID) -> acttype = ActType_t::LINEAR;
          
          cout << endl;
          cout << "Prediction all samples : " <<endl;
          layers.predict(input, SAMPLECOUNT, SAMPLEDIM, output, outsize);
          
          // show up the final score, to check the result consistency
          // first, setup the precision
          if(TypeIsDouble<T>::value) {
               std::cout.precision(std::numeric_limits<double>::digits10+1);
          } else {
               std::cout.precision(std::numeric_limits<float>::digits10+1);;
          }
          std::cout.setf( std::ios::fixed, std::ios::floatfield );        
          
          // then, select how many results will be shown.
          // if too many output, only show some in the beginning and some in the end
          if (outsize <= MAXSHOWRESULT){
               cout << endl << " Final score are :" <<endl;            
                for(int ii=0; ii<outsize; ii++){
                    cout << (output[ii]) << "  " ;
               }         
          } 
		else {
               cout << " Final score ( first " << MAXSHOWRESULT/2 << " records ):" <<endl;
               for(int ii=0; ii<(MAXSHOWRESULT/2); ii++){
                    cout << (output[ii]) << "  " ;
               }
               cout << endl << " Final score ( last " << MAXSHOWRESULT/2 << " records ):" <<endl;
               for(int ii=(outsize-MAXSHOWRESULT/2); ii<outsize; ii++){
                    cout << (output[ii]) << "  " ;
               }                         
          }
          cout << endl;        
          
     } 
	catch (...){
          if(bias!=NULL)       delete[] bias;
          if(bias_dims!=NULL)  delete[] bias_dims;
          if(data!=NULL)       delete[] data;
          if(data_dims!=NULL)  delete[] data_dims;  
          if(output!=NULL) delete[] output;          
          file.close();
     }

     // Free memory of allocated arraies.
     if(bias!=NULL)       delete[] bias;
     if(bias_dims!=NULL)  delete[] bias_dims;
     if(data!=NULL)       delete[] data;
     if(data_dims!=NULL)  delete[] data_dims;     
     if(output!=NULL) delete[] output;       
     file.close();
     return;
}


#endif
