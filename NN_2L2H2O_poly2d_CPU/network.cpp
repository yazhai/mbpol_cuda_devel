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

using namespace std;

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

struct Layer_t{
	string name;
	Type_t type;
	ActType_t acttype;
 	
	int inputs;     // number of input dimension
    	int outputs;    // number of output dimension
    	double ** weights; // weight matrix
    	double * bias; // bias vector 

	Layer_t * prev = nullptr;
	Layer_t * next = nullptr;
	
	Layer_t() :  name("Default_Layer"), weights(NULL), bias(NULL), inputs(0), 
		outputs(0), type(Type_t::UNINITIALIZED), acttype(ActType_t::NOACTIVIATION) {};


	Layer_t( string _name, int _inputs, int _outputs, 
          	double** _weights, double* _bias)
                  : inputs(_inputs), outputs(_outputs), type(Type_t::DENSE), acttype(ActType_t::NOACTIVIATION)
    	{     
        	name = _name ;
		size_t outputs_t = outputs, inputs_t = inputs;

		/*init Weight Matrix = Input x Output dimensions */
        	if(!init_mtx_in_mem<double>(weights,inputs_t,outputs_t)){
			cout<<"FAILED TO INITIALIZE MEMORY FOR WEIGHTS"<<endl;
		}
		for(int i=0;i<inputs;i++){
			copy(_weights[i],(_weights[i]+outputs),weights[i]);
		}

		/*init Bias Matrix = 1 x Output dimensions */
		bias = new double[outputs];
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
			clearMemo<double>(weights);
		}
	
		if(bias != NULL) delete [] bias;
	}

};


class network_t{

public:
		//N = number of dimers (second dimension of input matrix
	void fullyConnectedForward(const Layer_t & layer,
                          	int& input, int&output, int&N,
                          	double* srcData, double** dstData)
	{

		//store bias in dstData matrix(copy to each col) before computation
		//dstData is transposed to fully utilize rowMajor storage
		//maybe change to column vector x row vector to create matrix?
		for(int i=0;i<N;i++)
			for(int j=0;j<output;j++)
				dstData[i][j] = layer.bias[j];
	

		double ** temp = NULL;
		size_t output_t = output, N_t = N;
		transpose_mtx<double>(dstData,temp, N_t, output_t);
		
		
		cblas_dgemm(CblasRowMajor,CblasTrans,CblasTrans,output,N,input,1,layer.weights[0],output,srcData,input,1,*temp,N);
		
		transpose_mtx<double>(temp,dstData,output_t,N_t);
		
		clearMemo<double>(temp);
     	
    	} 

	void activationForward_TANH(const int & output,const int & N , double** srcData, double** dstData){
		double x;
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

class Layer_Net_t{
private: 
	
public:
	
	network_t neural_net;	//made public just for test purposes	
	
	Layer_t * root = nullptr;
	
	Layer_Net_t(){};
	
	~Layer_Net_t(){
		Layer_t* curr = nullptr;
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
          double ** & _weights, double * & _bias){
		if(root!=NULL){
			Layer_t * curr = root;
			while(curr->next){curr = curr->next;};
			curr->next = new Layer_t(_name,_inputs,_outputs,_weights,_bias);
			curr->next->prev = curr;
		}
		else{
			root = new Layer_t(_name, _inputs, _outputs, _weights, _bias);
		}
	}

	// Inserting an activiation layer by type (int)
     void insert_layer(string &_name, int _acttype){
          if (root!=NULL) {
               Layer_t* curr = root;
               while(curr->next) {curr = curr->next;};
               curr->next = new Layer_t(_name, _acttype);
               curr->next->prev = curr;
          }
		else{
			root = new Layer_t(_name,_acttype);
		}
	}

	void insert_layer(string &_name, ActType_t _acttype){
		if (root!=NULL) {
               Layer_t * curr = root;
               while(curr->next) {curr = curr->next;};
               curr->next = new Layer_t(_name, _acttype);
               curr->next->prev = curr;
          } 
		else {
               root = new Layer_t(_name, _acttype);
          }
     
     }     
};	


/*TESTER*/
int main(int argc, char** argv){ 
	int input=4, output=3, N = 2;
	size_t input_t = input, output_t = output, N_t = N;
	double ** testWeights = NULL;
	
	if(!init_mtx_in_mem<double>(testWeights,input_t,output_t))
		cout << "Test Weights failed to initialize!"<<endl;
	
	for(int i=0;i<input;i++){
		for(int j=0;j<output;j++){
			testWeights[i][j] = j+i;
		}
	}
	
	double * testBias = new double[3];
	
	testBias[0]= 0; testBias[1] =0; testBias[2] =0 ;

	Layer_Net_t * myNet = new Layer_Net_t();

	string name = "test";

	
	myNet->insert_layer(name,input,output,testWeights,testBias);
	
	double ** testIn = NULL;
	if(!init_mtx_in_mem<double>(testIn,N_t,input_t))
		cout << "Test In failed to initialize!"<<endl;

	for(int i=0;i<N;i++){
		for(int j=0;j<input;j++){
			testIn[i][j] = i*2+j;
		}
	}


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
	clearMemo<double>(testWeights);
	clearMemo<double>(testIn);
	clearMemo<double>(testOut);
	clearMemo<double>(testOut2);

	return 0;
	
}

