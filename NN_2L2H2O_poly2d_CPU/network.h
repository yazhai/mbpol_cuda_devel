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
#include "utility.h"
#include <H5Cpp.h>
#include <memory>
#include <iomanip>

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
 	
	int inputs;     // number of input dimension
    	int outputs;    // number of output dimension
    	T ** weights; // weight matrix
    	T * bias; // bias vector 

	Layer_t * prev = nullptr;
	Layer_t * next = nullptr;
	
	Layer_t() :  name("Default_Layer"), weights(NULL), bias(NULL), inputs(0), 
		outputs(0), type(Type_t::UNINITIALIZED), acttype(ActType_t::NOACTIVIATION) {};


	Layer_t( string _name, int _inputs, int _outputs, 
          	T** _weights, T* _bias)
                  : inputs(_inputs), outputs(_outputs), type(Type_t::DENSE), acttype(ActType_t::NOACTIVIATION)
    	{     
        	name = _name ;
		size_t outputs_t = outputs, inputs_t = inputs;

		/*init Weight Matrix = Input x Output dimensions */
        	if(!init_mtx_in_mem<T>(weights,inputs_t,outputs_t)){
			cout<<"FAILED TO INITIALIZE MEMORY FOR WEIGHTS"<<endl;
		}
		for(int i=0;i<inputs;i++){
			copy(_weights[i],(_weights[i]+outputs),weights[i]);
		}

		/*init Bias Matrix = 1 x Output dimensions */
		bias = new T[outputs];
		copy(_bias,_bias+outputs,bias);
	}

	// construct an activation layer, by integer 
	Layer_t(string _name, int _acttype)
                  : weights(NULL), bias(NULL),inputs(0), outputs(0), 
			type(Type_t::ACTIVIATION)
    {     
     	if (_acttype < int(ActType_t::MAX_ACTTYPE_VALUE) ) {
               acttype = static_cast<ActType_t>(_acttype);
          }
          name = _name;
    }

	//construct an activation layer by name
	Layer_t(string _name, ActType_t _acttype)
                  : weights(NULL),bias(NULL),inputs(0),outputs(0), 
                type(Type_t::ACTIVIATION)
    {      
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

template<typename T>
class network_t{

public:

	void fullyConnectedForward(const Layer_t<T> & layer,
                          	int& input, int&output, int&N,
                          	T* & srcData, T** & dstData)
	{
		output = layer.outputs;
		if(dstData != nullptr){
			clearMemo<T>(dstData);			
		}
		init_mtx_in_mem<T>(dstData,(size_t &)N,(size_t &)output);
//no transpose

		for(int i=0;i<N;i++){
			for(int j=0;j<output;j++){
				dstData[i][j] = 0;
				for(int k=0;k<input;k++){
					dstData[i][j] += srcData[input*i+k]*layer.weights[k][j];
				}
				dstData[i][j] +=layer.bias[j];
			}
			//cout<<dstData[i][j]
		}
		
		input = output;


//		Attempt at transposing before multiply


/*
		output = layer.outputs;		
		size_t output_t = output, N_t = N, input_t = input;

		T ** tempIn = NULL;
		T ** tempInT = NULL;
		T ** tempWeightsT = NULL;
		T ** tempWeights = NULL;
		T ** tempDestT = NULL;

		
		init_mtx_in_mem<T>(tempIn, N_t, input_t);  

		for(int i=0;i<N;i++)
			for(int j=0;j<input;j++)
				tempIn[i][j] = srcData[input*i+j];

		init_mtx_in_mem<T>(tempWeights,input_t,output_t);

		for(int i=0;i<input;i++){
			copy(layer.weights[i],layer.weights[i]+output,tempWeights[i]);
		}

		init_mtx_in_mem<T>(tempDestT,output_t,N_t);
		transpose_mtx<T>(tempIn,tempInT,N_t,input_t);
		transpose_mtx<T>(tempWeights,tempWeightsT,input_t,output_t);

		for(int i=0;i<output;i++){
			for(int j=0;j<N;j++){
				tempDestT[i][j] = 0;
				for(int k=0;k<input;k++){
					tempDestT[i][j] += tempWeightsT[i][k]*tempInT[k][j];
				}
				tempDestT[i][j] += layer.bias[i];
			}
		}

		transpose_mtx<T>(tempDestT,dstData,output_t,N_t);
		
		clearMemo<T>(tempIn);
		clearMemo<T>(tempInT);
		clearMemo<T>(tempWeightsT);
		clearMemo<T>(tempDestT);
		clearMemo<T>(tempIn);
		clearMemo<T>(tempWeights);
		input = output;

*/
    	
    	} 




	void activationForward_TANH(const int & output,const int & N , T** srcData, T** dstData){	
		
	
		T x;
		for(int i=0;i<N;i++){
			for(int j=0;j<output;j++){
				x = srcData[i][j];
				x = exp(2*x);
				x = (x-1)/(x+1);
				dstData[i][j] = x;
			}	
		}
	}
		

};

#if defined (_USE_GSL) || defined (_USE_MKL)
// Using cblas_dgemm if libraries are employed
template <>
void network_t<double>::fullyConnectedForward(const Layer_t<double> & layer,int& input, int&output, int&N,double* srcData, double** dstData);

template<>
void network_t<float>::fullyConnectedForward(const Layer_t<float> & layer,int& input, int&output, int&N,float* srcData, float** dstData);
#endif



template<typename T>
class Layer_Net_t{
private: 

	network_t<T> neural_net;
	
	void switchptr(T** & alpha, T** & bravo){
    		T** tmp;
          tmp = alpha;
          alpha = bravo;
          bravo = tmp;
          tmp = nullptr;
     }
	
public:
	
	//network_t<T> neural_net;	//made public just for test purposes	
	
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
	void insert_layer(string &_name, int _inputs, int _outputs, 
          T ** & _weights, T * & _bias){
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

	// Get layer ptr according to its index (start from 1 as 1st layer, 2 as seond layer ...)
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
             
			int input = _input;
			int N = _N;
             	int output = 1;

             	// two ptrs towards either alpha or bravo
             	// controlling from which the data is read
             	// and to which the result is written to 
             	T** srcDataPtr = nullptr; 
			T** dstDataPtr = nullptr;

			init_mtx_in_mem<T>(srcDataPtr,(size_t&)N,(size_t&)input);
			//for(int i=0;i<input;i++){
			//		copy(_inputData + input*i,_inputData + input*(i+1)-1,);
			//	}
			copy(_inputData,_inputData+input*N,srcDataPtr[0]);
			       
                                              
             	Layer_t<T>* curr = root;

            	do{
               	//cout << " Processing Layer : " << curr->name << endl;
               	if ( curr-> type == Type_t::DENSE ) { 
                    	// If it is a dense layer, we perform fully_connected forward 
                   		neural_net.fullyConnectedForward((*curr), input,output,N, *srcDataPtr, dstDataPtr);
                    	// Swith the origin/target memory array after the step
                    	switchptr(srcDataPtr, dstDataPtr);

                      
              		} else if (curr -> type == Type_t::ACTIVIATION){
                    	// If it is an activiation layer, perform corresponding activiation forwards
                    // In fact, activiation::linear = doing NOTHING 
                    	if (curr -> acttype == ActType_t::TANH){
                         	neural_net.activationForward_TANH(output,N, srcDataPtr, dstDataPtr);
                         	switchptr(srcDataPtr, dstDataPtr);
                    	} else if (curr->acttype == ActType_t::LINEAR) {    
                         	cout << "Linear Activation Layer Called" <<endl;
                    	} else {
						cout <<"Unknown Activation Type!"<<endl;
					}
              	 	} else {
                    	cout << "Unknown layer type!" <<endl;
               	}

        		} while(  (curr=curr->next) != NULL);
             
             	//cout << "Final score : " ;        
             	//printDeviceVector<T>(n*h*w, *srcDataPtr);
             
             	_outsize=input*N;
             	if(_outputData!=NULL){
                    delete[] _outputData;
             	}
             	_outputData = new T[_outsize];
             	//copy from srcDataPtr to outputData          
            	copy(*srcDataPtr,*srcDataPtr + _outsize,_outputData);
             
             
             	// Don't forget to release resource !!!
             	srcDataPtr = nullptr;
             	dstDataPtr = nullptr;
              
        
        }
        return;
	}
		    
};	

using namespace H5;
// tester function, including reading HDF5 file, creating layers, and making the prediction.
template <typename T>
void runtester(const char* filename, const char* checkchar, T* input){
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
               // for one single layer
               // layer's fullpath
               string layerpath = mkpath ( string(PATHTOMODEL),  *it ) ;
               
               // get this layer's dataset names
               vector<string> weights;
               weights = Read_Attr_Data_By_Seq(file,layerpath.c_str(), WEIGHTNAMES);
               
               
               cout << " Reading out layer data: " << *it << endl;
               for (auto it2 = weights.begin(); it2 != weights.end(); it2++){ 
                    // foe one data set
                    // dataset's path
                    string datasetPath = mkpath(layerpath,*it2) ;
                    
                    // check the dataset name's last character to see if this dataset is a Weight or a Bias
                    if ((*it2).compare(((*it2).length()-1),1, checkchar )==0){
                         // get out weight data
                         Read_Layer_Data_By_DatName<T> (file, datasetPath.c_str(), data, data_rank, data_dims); 
                    }else{
                         // get out bias data
                         Read_Layer_Data_By_DatName<T> (file, datasetPath.c_str(), bias, bias_rank, bias_dims);             
                    }
               }
               // When reading out a dense layer, a 2d weight matrix is obtained
               // Otherwise, it is a 0d matrix (null)
               if (data_rank==2){
                    cout << " Initialize dense layer : " << *it << endl;
	
				size_t inputs_t = data_dims[0], outputs_t = data_dims[1];
				T** weights_Matrix;
				init_mtx_in_mem<T>(weights_Matrix,(size_t&)data_dims[0],(size_t&)data_dims[1]);							//TODO: bad solution
				//for(int i=0;i<data_dims[0];i++){
				//	copy(data + data_dims[1]*i,data+data_dims[1]*(i+1)-1,weights_Matrix[i]);
				//}
				copy(data,data+data_dims[0]*data_dims[1],weights_Matrix[0]);
                    

				layers.insert_layer(*it, data_dims[0], data_dims[1], weights_Matrix, bias);			
				
				clearMemo<T>(weights_Matrix);
                    
				data_rank=0;
                    bias_rank=0;
               } else {
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
          } else {
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
          
          
     } catch (...){
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
