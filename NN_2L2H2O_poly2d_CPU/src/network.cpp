#include <iostream>
#include <sstream>
#include <fstream>
#include <stdlib.h>
#include <string>   
#include <limits>
#include <vector>
#include <math.h>
#include <H5Cpp.h>
#include <memory>
#include <iomanip>

#include "utility.h"
#include "network.h"
#include "timestamps.h"
#include "readhdf5.hpp"

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
using namespace MBbpnnPlugin;

//***********************************************************************************
// Structure of Network:
//          dimensions: N = number of samples or clusters, input = sample input Dimension, 
//               output = sample output dimension
//          Layer Weights: Inputed as  input x output dimensional array
//                         -stored as output x input dimensional array(transpose(weights))
//          Layer Bias:     Inputed as 1xoutput dimensional array
//                              extended to Nxoutput dimensional array for computation
//          Layer Input:     Inputed as N x input dimensional array
//          
//          At each Dense Layer:
//          Output = Input*Weights + Bias
//          
//          OR: Output = transpose(Weights) * transpose(Input) + transpose(Bias)
//
//          This second method is used in the code in order to take advantage of 
//          rowMajor memory storage.
//
//          Weights Matrix is transposed when layer is initialized, while Input data
//               is transposed in the predict method of NN_t (before forward 
//               propagation through the network begins). Transposing Bias is trivial.
//***********************************************************************************



/* Layer_t struct method definitions */

//Default constructor
template<typename T>
Layer_t<T>::Layer_t() :  name("Default_Layer"), weights(NULL), bias(NULL), inputs(0), 
          outputs(0), type(Type_t::UNINITIALIZED), acttype(ActType_t::NOACTIVIATION) {};

//Dense Layer Constructor
template<typename T>
Layer_t<T>::Layer_t( string _name, size_t _inputs, size_t _outputs, 
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

//Activation Layer Constructor(using integer as type)
template<typename T>
Layer_t<T>::Layer_t(string _name, int _acttype, size_t outputs)
                    : weights(NULL), bias(NULL),inputs(outputs), outputs(outputs), 
          type(Type_t::ACTIVIATION){
     
     if (_acttype < int(ActType_t::MAX_ACTTYPE_VALUE) ) {
               acttype = static_cast<ActType_t>(_acttype);
          }
     name = _name;
}

//Activation Layer Constructor(using enum ActType_t)
template<typename T>
Layer_t<T>::Layer_t(string _name, ActType_t _acttype, size_t outputs)
                    : weights(NULL),bias(NULL),inputs(outputs),outputs(outputs), 
               type(Type_t::ACTIVIATION){
     
          acttype = _acttype;
          name = _name;
     } 

//Destructor
template<typename T>
Layer_t<T>::~Layer_t(){
     if(weights != NULL){
          clearMemo<T>(weights);
     }
     if(bias != NULL) delete [] bias;
}



/* Functional_t class method definitions */

//non-cblas implementation of fully Connected Forward Propogation.
//Weights matrix and srcData are already transposed for optimal computation
template<typename T>
void Functional_t<T>::fullyConnectedForward(const Layer_t<T> & layer,
                              const size_t N,
                              T* & srcData, T** & dstData){

     size_t output = layer.outputs;
     size_t input = layer.inputs;

     //create space for output
     if(dstData != nullptr){
          clearMemo<T>(dstData);     
     }
     init_mtx_in_mem<T>(dstData,output,N);

     //Conduct Matrix Multiplication, use N as inner-most loops, because it
     //will be the biggest!
     T** wt = layer.weights;
     for(int i=0;i<output;i++){
          for(int j=0;j<input;j++){
               #ifdef _OPENMP 
               #pragma omp parallel for simd shared(dstData,srcData,wt)
               #endif 
               for (int k=0;k<N;k++){
                    dstData[i][k] += wt[i][j]*srcData[N*j+k];
               }
          }
     }
     
     //Conduct Bias Addition, se N as inn-ermost loop, beacuse it will be
     //the biggest!
     T* bs = layer.bias;
     for(int i=0;i< output;i++){
          #ifdef _OPENMP
          #pragma omp parallel for simd shared(dstData,bs)
          #endif 
          for(int j=0;j<N;j++){
               dstData[i][j] += bs[i];
          }
     }

} 

//non-cblas implementaiton of forward propogation activation function(TANH)
template <typename T>
void Functional_t<T>::activationForward_TANH(Layer_t<T> & layer,  const size_t N , T* srcData, T** & dstData, bool ifgrd){     

     size_t output = layer.outputs;

     //create space for output
     if(dstData != nullptr){
          clearMemo<T>(dstData);     
     }
     init_mtx_in_mem<T>(dstData,output,N);
     
     if (ifgrd) {
          init_mtx_in_mem<T>(layer.weights, output, N);
          T** grd_tmp = layer.weights;
          //complete faster TANH computation
          for(int i=0;i<output;i++){
               #ifdef _OPENMP
               #pragma omp parallel for simd shared(srcData,dstData, grd_tmp)
               #endif 
               for(int j=0;j<N;j++){
                    T x = exp( srcData[i*N + j] *2 );
                    dstData[i][j]  = (x-1)/(x+1);
                    grd_tmp[i][j] = (4*x) / ( (x+1) * (x+1) );
               }     
          }
     } else{
          //complete faster TANH computation
          for(int i=0;i<output;i++){
               #ifdef _OPENMP
               #pragma omp parallel for simd shared(srcData,dstData)
               #endif 
               for(int j=0;j<N;j++){
                    T x = exp( srcData[i*N + j] *2 );
                    dstData[i][j]  = (x-1)/(x+1);
               }     
          }          
     }
}     


//non-cblas implementation of fully Connected Backward Propogation.
template<typename T>
void Functional_t<T>::fullyConnectedBackward(const Layer_t<T> & layer,
                              const size_t N,
                              T* & srcData, T** & dstData){
     
     //backward matrix dimension
     size_t output = layer.inputs;
     size_t input = layer.outputs;

     //create space for output
     if(dstData != nullptr){
          clearMemo<T>(dstData);     
     }
     init_mtx_in_mem<T>(dstData,output,N);

     //Conduct Matrix Multiplication, use N as inner-most loops, because it
     //will be the biggest!
     T** wt = layer.weights;
     for(int j=0;j< output;j++){
          for(int i=0;i<input;i++){
               #ifdef _OPENMP
               #pragma omp parallel for simd shared(dstData,srcData,wt)
               #endif 
               for (int k=0;k<N;k++){
                    dstData[j][k] += wt[i][j]*srcData[N*i+k];
               }
          }
     }
} 


//non-cblas implementaiton of forward propogation activation function(TANH)
template <typename T>
void Functional_t<T>::activationBackward_TANH(const Layer_t<T> & layer,  const size_t N , T* srcData, T** & dstData){     

     size_t output = layer.outputs;

     //create space for output
     if(dstData != nullptr){
          clearMemo<T>(dstData);     
     }
     init_mtx_in_mem<T>(dstData,output,N);
     
     //complete faster TANH computation
     T** wt = layer.weights;
     for(int i=0;i<output;i++){
          #ifdef _OPENMP
          #pragma omp parallel for simd shared(srcData,dstData,wt)
          #endif 
          for(int j=0;j<N;j++){
               dstData[i][j] = srcData[i*N + j] * wt[i][j];
          }     
     }
}  


//cblas implementations of double and single precision forward dense layer
//Input and layer.Weights already transposed for optimal row-major computation.
#if defined (_USE_GSL) || defined (_USE_MKL)
template <>
void Functional_t<double>::fullyConnectedForward(const Layer_t<double> & layer,
                               const size_t & N,
                               double* & srcData, double** & dstData)
     {
          output = layer.outputs;     //update the dimensions of the layer

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
void Functional_t<float>::fullyConnectedForward(const Layer_t<float> & layer,
                               const size_t & N,
                               float* & srcData, float** & dstData)
     {

          output = layer.outputs;     //update the dimensions of the layer
          
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

template<>
void Functional_t<double>::fullyConnectedBackward(const Layer_t<double> & layer,
                               const size_t N,
                               double* & srcData, double** & dstData)
     {
          output = layer.inputs;     //update the dimensions of the layer
          input = layer.outputs;

          //create space for output
          if(dstData != nullptr)
               clearMemo<double>(dstData);     
          init_mtx_in_mem<double>(dstData,output,N);

          //compute Weights(^T)[output][input] x Input[input][N] 
          cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,output,N,input,1.0,layer.weights[0],input,srcData,N,1,*dstData,N);
         }; 

template<>
void Functional_t<float>::fullyConnectedBackward(const Layer_t<float> & layer,
                               const size_t N,
                               float* & srcData, float** & dstData)
     {
          output = layer.inputs;     //update the dimensions of the layer
          input = layer.outputs;
          
          //create space for output
          if(dstData != nullptr)
               clearMemo<float>(dstData);     
          init_mtx_in_mem<float>(dstData,output,N);

          //compute Weights(^T)[output][input] x Input[input][N] + Bias[output][N]
          cblas_sgemm(CblasRowMajor,CblasTrans,CblasNoTrans,output,N,input,1.0,layer.weights[0],input,srcData,N,1,*dstData,N);
          
     };

#endif



/* NN_t class method definitions */

//helper function: switch two pointers to pointers
template <typename T>
void NN_t<T>::switchptr(T** & alpha, T** & bravo){ 
     T** tmp;
     tmp = alpha;
     alpha = bravo;
     bravo = tmp;
     tmp = nullptr;
}

//Delete all layers, from end to root.
template <typename T>
NN_t<T>::~NN_t(){
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
template <typename T>
void NN_t<T>::insert_layer(string &_name, size_t _inputs, size_t _outputs, 
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
template <typename T>
 void NN_t<T>::insert_layer(string &_name, int _acttype, size_t _outputs){
          if (root!=NULL) {
               Layer_t<T>* curr = root;
               while(curr->next) {curr = curr->next;};
               curr->next = new Layer_t<T>(_name, _acttype, _outputs);
               curr->next->prev = curr;
          }
     else{
          root = new Layer_t<T>(_name,_acttype,_outputs);
     }
}

//Inserting an activation layer by type(enum)
template <typename T>
void NN_t<T>::insert_layer(string &_name, ActType_t _acttype, size_t _outputs){
     if (root!=NULL) {
               Layer_t<T> * curr = root;
               while(curr->next) {curr = curr->next;};
               curr->next = new Layer_t<T>(_name, _acttype, _outputs);
               curr->next->prev = curr;
          } 
     else {
               root = new Layer_t<T>(_name, _acttype, _outputs);
          }
     
}

// Get layer ptr according to its index (start from 1 as 1st layer, 2 as second layer ...)
template <typename T>
Layer_t<T>* NN_t<T>::get_layer_by_seq(int _n){
     Layer_t<T>* curr=root;
     int i = 1;
     
     while( (curr->next != NULL)  && (i<_n) ){
          curr = curr->next;
          i++ ;
     };
     return curr;
} 

//Move through network and make prediction based on all layers.     
template <typename T>
void NN_t<T>::predict(T* _inputData, size_t input, size_t N, T* & _outputData){
     if (root != NULL) {
   
          //two pointers used to store and recieve data(switch between them)
          T** srcDataPtr = nullptr; 
          T** dstDataPtr = nullptr;

          //init srcDataPtr to point to tranpose of input data
          init_mtx_in_mem<T>(srcDataPtr, input, N);
          copy(_inputData,_inputData+input*N,srcDataPtr[0]);
                                                  
          Layer_t<T>* curr = root;
          do{
               // cout<<curr->name<<endl;
               //DENSE LAYER PROPOGATION
               if ( curr-> type == Type_t::DENSE ) { 
                    
                         // If it is a dense layer, we perform fully_connected forward 
                         neural_net.fullyConnectedForward((*curr), N, *srcDataPtr, dstDataPtr);
                    //note: inside fullyConnectedForward, output is updated, and input=output for next layer use.

                         switchptr(srcDataPtr, dstDataPtr);
                    } 
               //ACTIVATION LAYER PROPOGATION
               else if (curr -> type == Type_t::ACTIVIATION){
                         // If it is an activiation layer, perform corresponding activiation forwards
                         if (curr -> acttype == ActType_t::TANH){
                              neural_net.activationForward_TANH((*curr),N, *srcDataPtr, dstDataPtr);
                              switchptr(srcDataPtr, dstDataPtr);
                         } 
                    else if (curr->acttype == ActType_t::LINEAR) {    
     
                         } 
                    else {
                         cout <<"Unknown Activation Type!"<<endl;
                    }
                    } 
               else {
                         cout << "Unknown layer type!" <<endl;
               }

          } 
          
          while(  (curr->next!= NULL) && (curr = curr->next) );
               
          //create space for output Data
          if(_outputData!=NULL){
               delete[] _outputData;
          }
          _outputData = new T[N*curr->outputs];

          //copy from srcDataPtr to outputData          
          copy(*srcDataPtr,*srcDataPtr + N*curr->outputs,_outputData);
                    
          //Release Resources
          clearMemo<T>(srcDataPtr);
          clearMemo<T>(dstDataPtr);
          srcDataPtr = nullptr;
          dstDataPtr = nullptr;  
     }

     return;
}      



//Move through network and make prediction based on all layers.     
template <typename T>
void NN_t<T>::predict_and_getgrad(T* _inputData, size_t input, size_t N, T* & _outputData, T* & _grdData){
     if (root != NULL) {
   
          //two pointers used to store and recieve data(switch between them)
          T** srcDataPtr = nullptr; 
          T** dstDataPtr = nullptr;

          //init srcDataPtr to point to tranpose of input data
          init_mtx_in_mem<T>(srcDataPtr, input, N);
          copy(_inputData,_inputData+input*N,srcDataPtr[0]);
                                                  
          Layer_t<T>* curr = root;
          do{
               // cout<<curr->name<<endl;
               //DENSE LAYER PROPOGATION
               if ( curr-> type == Type_t::DENSE ) { 
                    
                    // If it is a dense layer, we perform fully_connected forward 
                    neural_net.fullyConnectedForward((*curr), N, *srcDataPtr, dstDataPtr);
                    //note: inside fullyConnectedForward, output is updated, and input=output for next layer use.

                    switchptr(srcDataPtr, dstDataPtr);
               } 
               //ACTIVATION LAYER PROPOGATION
               else if (curr -> type == Type_t::ACTIVIATION){
                         // If it is an activiation layer, perform corresponding activiation forwards
                         if (curr -> acttype == ActType_t::TANH){
                              neural_net.activationForward_TANH((*curr),N, *srcDataPtr, dstDataPtr, true);
                              switchptr(srcDataPtr, dstDataPtr);
                         } 
                    else if (curr->acttype == ActType_t::LINEAR) {    
     
                         } 
                    else {
                         cout <<"Unknown Activation Type!"<<endl;
                    }
                    } 
               else {
                         cout << "Unknown layer type!" <<endl;
               }

          } while(  (curr->next!= NULL) && (curr = curr->next) ) ;
               
          //create space for output Data
          if(_outputData!=NULL){
               delete[] _outputData;
          }
          _outputData = new T[N*curr->outputs];

          //copy from srcDataPtr to outputData        
          copy(*srcDataPtr,*srcDataPtr + N*curr->outputs,_outputData);


          // do the backwards immediately 
          // init srcDataPtr to 1
          std::fill_n(*srcDataPtr, N*curr->outputs, 1.0);

          while( true ){
               //DENSE LAYER PROPOGATION
               if ( curr-> type == Type_t::DENSE ) { 
                    
                    // If it is a dense layer, we perform fully_connected backward 
                    neural_net.fullyConnectedBackward((*curr), N, *srcDataPtr, dstDataPtr);
                    //note: inside fullyConnectedForward, output is updated, and input=output for next layer use.

                    switchptr(srcDataPtr, dstDataPtr);
               } 
               //ACTIVATION LAYER PROPOGATION
               else if (curr -> type == Type_t::ACTIVIATION){
                    // If it is an activiation layer, perform corresponding activiation forwards
                    if (curr -> acttype == ActType_t::TANH){
                         neural_net.activationBackward_TANH((*curr),N, *srcDataPtr, dstDataPtr);
                         switchptr(srcDataPtr, dstDataPtr);
                    } 
                    else if (curr->acttype == ActType_t::LINEAR) {    
     
                         } 
                    else {
                         cout <<"Unknown Activation Type!"<<endl;
                    }
               } 
               else {
                         cout << "Unknown layer type!" <<endl;
               };

               if (curr->prev != NULL ) {
                    curr = curr->prev;
               } else{
                    break;
               };

          };

          //create space for grd Data
          if( _grdData!=NULL){
               delete[] _grdData;
          }
          _grdData = new T[curr->inputs * N];

          //copy from srcDataPtr to outputData
          copy(*srcDataPtr,*srcDataPtr + curr->inputs * N, _grdData);

          //Release Resources
          clearMemo<T>(srcDataPtr);
          clearMemo<T>(dstDataPtr);
          srcDataPtr = nullptr;
          dstDataPtr = nullptr;  
     }

     return;
}



//Move backwards through the network and return the gradient
template <typename T>
void NN_t<T>::backward(T* _inputData, size_t input, size_t N, T* & _outputData){
          if (root != NULL) {
               Layer_t<T>* curr = root;
               while (curr->next != nullptr) curr = curr->next ;

               //two pointers used to store and recieve data(switch between them)
               T** srcDataPtr = nullptr; 
               T** dstDataPtr = nullptr;


               //init srcDataPtr to point to input data
               init_mtx_in_mem<T>(srcDataPtr, input, N);
               if (_inputData != nullptr ){
                    copy(_inputData,_inputData+input*N,srcDataPtr[0]);
               } else {
                    std::fill_n(*srcDataPtr, N*input, 1.0);
               };
               


               

               while( true ){
                    //DENSE LAYER PROPOGATION
                    if ( curr-> type == Type_t::DENSE ) { 
                         
                              // If it is a dense layer, we perform fully_connected backward 
                              neural_net.fullyConnectedBackward((*curr), N, *srcDataPtr, dstDataPtr);
                         //note: inside fullyConnectedForward, output is updated, and input=output for next layer use.

                              switchptr(srcDataPtr, dstDataPtr);
                         } 
                    //ACTIVATION LAYER PROPOGATION
                    else if (curr -> type == Type_t::ACTIVIATION){
                         // If it is an activiation layer, perform corresponding activiation forwards
                         if (curr -> acttype == ActType_t::TANH){
                              neural_net.activationBackward_TANH((*curr),N, *srcDataPtr, dstDataPtr);
                              switchptr(srcDataPtr, dstDataPtr);
                         } 
                         else if (curr->acttype == ActType_t::LINEAR) {    
          
                              } 
                         else {
                              cout <<"Unknown Activation Type!"<<endl;
                         }
                    } 
                    else {
                              cout << "Unknown layer type!" <<endl;
                    };

                    if (curr->prev != NULL ) {
                         curr = curr->prev;
                    } else{
                         break;
                    };

               };
               
          //create space for output Data
          if(_outputData!=NULL){
               delete[] _outputData;
          }
          _outputData = new T[curr->inputs * N];

          //copy from srcDataPtr to outputData          
          copy(*srcDataPtr,*srcDataPtr + curr->inputs * N, _outputData);
                         
          //Release Resources
          clearMemo<T>(srcDataPtr);
          clearMemo<T>(dstDataPtr);
          srcDataPtr = nullptr;
          dstDataPtr = nullptr;
     }

     return;
}    




template<typename T>
allNN_t<T>::allNN_t():  nets(nullptr), numNetworks(0) {};

template<typename T>
allNN_t<T>::allNN_t(size_t _numNetworks, const char * filename, const char * checkchar):nets(nullptr) {
     init_allNNs(_numNetworks, filename,  checkchar);
}


template<typename T>
void allNN_t<T>::init_allNNs(size_t _numNetworks, const char * filename, const char * checkchar){
      numNetworks = _numNetworks;

      nets = new NN_t<T>[numNetworks];
      H5::H5File file(filename,H5F_ACC_RDONLY);

      hsize_t data_rank=0;
      hsize_t* data_dims = nullptr;
      T* data = nullptr;     
      hsize_t bias_rank=0;
      hsize_t* bias_dims = nullptr;
      T* bias = nullptr;

      vector<string> sequence;
      sequence = Read_Attr_Data_By_Seq(file,PATHTOMODEL, LAYERNAMES); 

      string networkNum, networkName;
      int currentNetNum = 0;


      
      networkNum = std::to_string(currentNetNum +1);
      networkName = "sequential_"+ networkNum;

      int layerID=1; 
      for(auto it=sequence.begin();it!=sequence.end();it++){
            string seqPath = mkpath ( string(PATHTOMODEL),  *it );
            if(*it == networkName){
                  // get this layer's dataset names(weights and bias)
                  vector<string> weights;
                  weights = Read_Attr_Data_By_Seq(file,seqPath.c_str(), WEIGHTNAMES);
               //    cout<<*it<<endl;

                  cerr << " Reading out data: " << *it << endl;
                  for (auto it2 = weights.begin(); it2 != weights.end(); it2++){ 
                        // for one data set get path
                        string datasetPath = mkpath(seqPath,*it2) ;
                    
                        //Dense Layer Name
                        string DLname = (*it2).substr(0,7);

                        // check the dataset name's last character to see if this dataset is a Weight or a Bias
                         if ((*it2).compare(((*it2).length()-1),1, checkchar )==0){
                              
                              
                              // get out weight data
                              Read_Layer_Data_By_DatName<T> (file, datasetPath.c_str(), data, data_rank, data_dims); 
                         }
                         else{
                              // get out bias data
                              Read_Layer_Data_By_DatName<T> (file, datasetPath.c_str(), bias, bias_rank, bias_dims);\
                              cerr << " Initialize layer : " << DLname << endl;
                         
                              nets[currentNetNum].insert_layer(DLname, data_dims[0], data_dims[1], data, bias);

                              cerr << " Layer " << DLname << "  is initialized. " <<endl <<endl; 


                              //insert tanh activation layer afterwards
                              string actName = "Activation_" + to_string(layerID/2);
                              
                              cerr << " Initialize layer : " << actName << endl;
                              nets[currentNetNum].insert_layer(actName, ActType_t::TANH,data_dims[1]);
                              cerr << " Layer " << actName << "  is initialized. " <<endl <<endl;                                    
                    
                              //reset values for next loop
                              data_rank=0;
                              bias_rank=0; 
                        }
                        
                        layerID++;

                  } 
                  //make last layer activation type linear
                  nets[currentNetNum].get_layer_by_seq(layerID) -> acttype=ActType_t::LINEAR;
                  cerr<<"Changing Layer "<<nets[currentNetNum].get_layer_by_seq(layerID)->name<<" to Linear Activation"<<endl;
                  cerr << "Inserting Layers " << *it<< " Finished!"  <<endl;

                  currentNetNum++;
                  
                  networkNum = std::to_string(currentNetNum +1);
                  networkName = "sequential_"+ networkNum;   
            }
      }  

       // Free memory of allocated arrays.
     if(bias!=NULL)       delete[] bias;
     if(bias_dims!=NULL)  delete[] bias_dims;
     if(data!=NULL)       delete[] data;
     if(data_dims!=NULL)  delete[] data_dims;      
     file.close();
     return;                             
}


template<typename T>
allNN_t<T>::~allNN_t(){
     if (nets != nullptr) delete [] nets;
}


//Instantiate all non-template class/struct/method of specific type
template struct Layer_t<double>;
template struct Layer_t<float>;

template class Functional_t<double>;
template class Functional_t<float>;

template class NN_t<double>;
template class NN_t<float>;

template struct allNN_t<double>;
template struct allNN_t<float>;




