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
                          	T* srcData, T** dstData)
	{

//no transpose

		for(int i=0;i<N;i++){
			for(int j=0;j<output;j++){
				dstData[i][j] = 0;
				for(int k=0;k<input;k++){
					dstData[i][j] += srcData[input*i+k]*layer.weights[k][j];
				}
				dstData[i][j] +=layer.bias[j];
			}
		}


//		Attempt at transposing before multiply


/*
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
	
public:
	
	network_t<T> neural_net;	//made public just for test purposes	
	
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
};	



#endif
