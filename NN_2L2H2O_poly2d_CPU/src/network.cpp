#include <iostream>
#include <sstream>
#include <fstream>
#include <stdlib.h>
#include <string>   
#include <limits>
#include <vector>
#include <math.h>
#include "H5Cpp.h"
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


//#define EUNIT (6.0)/(.0433634) //energy conversion factor 
#define EUNIT 1

using namespace std;

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
//               is transposed in the predict method of Layer_Net_t (before forward 
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
                    : weights(NULL), bias(NULL),inputs(0), outputs(outputs), 
          type(Type_t::ACTIVIATION){
     
     if (_acttype < int(ActType_t::MAX_ACTTYPE_VALUE) ) {
               acttype = static_cast<ActType_t>(_acttype);
          }
     name = _name;
}

//Activation Layer Constructor(using enum ActType_t)
template<typename T>
Layer_t<T>::Layer_t(string _name, ActType_t _acttype, size_t outputs)
                    : weights(NULL),bias(NULL),inputs(0),outputs(outputs), 
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



/* network_t class method definitions */

//non-cblas implementation of fully Connected Forward Propogation.
//Weights matrix and srcData are already transposed for optimal computation
template<typename T>
void network_t<T>::fullyConnectedForward(const Layer_t<T> & layer,
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
     int k=0;
     for(int i=0;i<output;i++){
          for(int j=0;j<input;j++){
               #ifdef _OPENMP 
               #pragma omp parallel for simd shared(dstData,srcData,layer)
               #endif 
               for (k=0;k<N;k++){
                    dstData[i][k] += layer.weights[i][j]*srcData[N*j+k];
               }
          }
     }
     
     //Conduct Bias Addition, se N as inn-ermost loop, beacuse it will be
     //the biggest!
     for(int i=0;i< output;i++){
          #ifdef _OPENMP
          #pragma omp parallel for simd shared(dstData,layer)
          #endif 
          for(int j=0;j<N;j++){
               dstData[i][j] += layer.bias[i];
          }
     }

} 

//non-cblas implementaiton of forward propogation activation function(TANH)
template <typename T>
void network_t<T>::activationForward_TANH(const Layer_t<T> & layer,  const size_t N , T* srcData, T** & dstData){     

     size_t output = layer.outputs;

     //create space for output
     if(dstData != nullptr){
          clearMemo<T>(dstData);     
     }
     init_mtx_in_mem<T>(dstData,output,N);
     
     //complete faster TANH computation
     T x;
     for(int i=0;i<output;i++){
          #ifdef _OPENMP
          #pragma omp parallel for simd shared(srcData,dstData,x)
          #endif 
          for(int j=0;j<N;j++){
               x = srcData[i*N + j];
               x = exp(2*x);
               x = (x-1)/(x+1);
               dstData[i][j] = x;
          }     
     }
}     


//non-cblas implementation of fully Connected Backward Propogation.
template<typename T>
void network_t<T>::fullyConnectedBackward(const Layer_t<T> & layer,
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
     int k=0;
     for(int j=0;j< output;j++){
          for(int i=0;i<input;i++){
               #ifdef _OPENMP
               #pragma omp parallel for simd shared(dstData,srcData,layer)
               #endif 
               for (k=0;k<N;k++){
                    dstData[j][k] += layer.weights[i][j]*srcData[N*i+k];
               }
          }
     }
} 


//non-cblas implementaiton of forward propogation activation function(TANH)
template <typename T>
void network_t<T>::activationBackward_TANH(const Layer_t<T> & layer,  const size_t N , T* srcData, T** & dstData){     

     size_t output = layer.outputs;

     //create space for output
     if(dstData != nullptr){
          clearMemo<T>(dstData);     
     }
     init_mtx_in_mem<T>(dstData,output,N);
     
     //complete faster TANH computation
     T x;
     for(int i=0;i<output;i++){
          #ifdef _OPENMP
          #pragma omp parallel for simd shared(srcData,dstData,x)
          #endif 
          for(int j=0;j<N;j++){
               x = srcData[i*N + j];
               x = exp(2*x);
               x = (4*x) / (x+1) / (x+1);
               dstData[i][j] = x;
          }     
     }
}  


//cblas implementations of double and single precision forward dense layer
//Input and layer.Weights already transposed for optimal row-major computation.
#if defined (_USE_GSL) || defined (_USE_MKL)
template <>
void network_t<double>::fullyConnectedForward(const Layer_t<double> & layer,
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
void network_t<float>::fullyConnectedForward(const Layer_t<float> & layer,
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
void network_t<double>::fullyConnectedBackward(const Layer_t<double> & layer,
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
void network_t<float>::fullyConnectedBackward(const Layer_t<float> & layer,
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



/* Layer_Net_t class method definitions */

//helper function: switch two pointers to pointers
template <typename T>
void Layer_Net_t<T>::switchptr(T** & alpha, T** & bravo){ 
     T** tmp;
     tmp = alpha;
     alpha = bravo;
     bravo = tmp;
     tmp = nullptr;
}

//Delete all layers, from end to root.
template <typename T>
Layer_Net_t<T>::~Layer_Net_t(){
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
void Layer_Net_t<T>::insert_layer(string &_name, size_t _inputs, size_t _outputs, 
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
 void Layer_Net_t<T>::insert_layer(string &_name, int _acttype, size_t _outputs){
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
void Layer_Net_t<T>::insert_layer(string &_name, ActType_t _acttype, size_t _outputs){
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
Layer_t<T>* Layer_Net_t<T>::get_layer_by_seq(int _n){
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
void Layer_Net_t<T>::predict(T* _inputData, int _N, int _input, T* & _outputData){
          if (root != NULL) {
             
          size_t input = _input;
          size_t N = _N;
    
          //two pointers used to store and recieve data(switch between them)
          T** srcDataPtr = nullptr; 
          T** dstDataPtr = nullptr;

          //init srcDataPtr to point to tranpose of input data
          T** temp;
          init_mtx_in_mem<T>(temp,N,input);
          copy(_inputData,_inputData+input*N,temp[0]);
          
          cout<<"Predict Input: "<<input<<" Predict N: "<<N<<endl;
          
          transpose_mtx<T>(srcDataPtr, temp, N, input);

          clearMemo<T>(temp);
                                                  
               Layer_t<T>* curr = root;
               do{
               cout<<curr->name<<endl;
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
          while(  (curr=curr->next) != NULL);
               
          //create space for output Data
               if(_outputData!=NULL){
                    delete[] _outputData;
               }
               _outputData = new T[N];
     
               //copy from srcDataPtr to outputData          
               copy(*srcDataPtr,*srcDataPtr + N,_outputData);
                         
               //Release Resources
          clearMemo<T>(srcDataPtr);
          clearMemo<T>(dstDataPtr);
          srcDataPtr = nullptr;
               dstDataPtr = nullptr;  
     }

     return;
}      





/* TESTER FUNCTION 
     Input:    filename -- weights and bias datafile -- defined in "fullTester.cpp"
               checkchar - character used in processing datafile to differentiate between weights and biases -- defined in "fullTester.cpp"
               input -- 2-d array of Gfn outputs (numAtoms x (N*sampleDim[i]))
               numAtoms -- number of atoms to be proccessed (first dimension of input array)
               N  --  the number of samples for each atom (second dimension of input array)
               sampleDim -- a 1 x numAtoms sized array containing information for the number of inputs per sample
     Result:
               Printing of first/last 10 scores for each atom
               Printing of first/last 10 scores for the final output(summation of all atoms)
               Store all scores of each atom in file -- "NN_final.out"
               Store final output score(summation of all atoms scores) in file -- "my_y_pred.txt"
               The above file can be compared with file "y_pred.txt" which contains outputs from python implementation
*/
template <typename T>
void runtester(const char* filename, const char* checkchar, T** input, size_t N,T* cutoffs, 
           const std::vector<idx_t> & typesList, const std::vector<size_t> & inputSizePerType){

     size_t numAtoms = typesList.size();
     using namespace H5;
     
     ofstream outputFile;
     string filePath = "NN_final.out";
     // initialize memory for rank, dims, and data
    hsize_t data_rank=0;
    hsize_t* data_dims = nullptr;
    T* data = nullptr;     
     
    hsize_t bias_rank=0;
    hsize_t* bias_dims = nullptr;
    T* bias = nullptr;
     
    Layer_Net_t<T> layers_1;
     Layer_Net_t<T> layers_2;

     Layer_Net_t<T> * currentNet = nullptr;
     
     // reserver for results
     //unsigned long int outsize = 0; 
     T* output = nullptr;
     T* finalOutput = nullptr;    
     
     // Open HDF5 file handle, read only
     H5File file(filename,H5F_ACC_RDONLY);
     
     try{     
          // Get saved layer names
          vector<string> sequence;
          sequence = Read_Attr_Data_By_Seq(file,PATHTOMODEL, LAYERNAMES); 
          int layerID=1;          
          for (auto it=sequence.begin();it!=sequence.end();it++) {
     
               // for one single layer get path
               string seqPath = mkpath ( string(PATHTOMODEL),  *it ) ;
               
               // get this layer's dataset names(weights and bias)
               vector<string> weights;
               weights = Read_Attr_Data_By_Seq(file,seqPath.c_str(), WEIGHTNAMES);
               cout<<*it<<endl;
               if(weights.size() == 0){
                    //this layer is not just an input, and not really a layer
                    continue;
               }
               else if(*it == "sequential_1"){
                    currentNet = & layers_1;
               }
               else{
                    currentNet = & layers_2;

               }
                    
                    cout << " Reading out data: " << *it << endl;
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
                              cout << " Initialize layer : " << DLname << endl;
                         
                              (*currentNet).insert_layer(DLname, data_dims[0], data_dims[1], data, bias);

                              cout << " Layer " << DLname << "  is initialized. " <<endl <<endl; 


                              //insert tanh activation layer afterwards
                              string actName = "Activation_" + to_string(layerID/2);
                              
                              cout << " Initialize layer : " << actName << endl;
                              (*currentNet).insert_layer(actName, ActType_t::TANH,data_dims[1]);
                              cout << " Layer " << actName << "  is initialized. " <<endl <<endl;                                    
                    
                              //reset values for next loop
                              data_rank=0;
                              bias_rank=0;        
                              
                         }
                         layerID++;
                    }
                    
                    //make last layer activation type linear
                    (*currentNet).get_layer_by_seq(layerID) -> acttype=ActType_t::LINEAR;
                    cout<<"Changing Layer "<<(*currentNet).get_layer_by_seq(layerID)->name<<" to Linear Activation"<<endl;
                    cout << "Inserting Layers " << *it<< " Finished!"  <<endl;
              
               }
          
          cout << endl;
          cout << "Prediction all samples : " <<endl;


          //delete current content of output file
          outputFile.open(filePath,ofstream::out|ofstream::trunc);
          outputFile.close();
          //double * finalOutput = nullptr;
          
          for(int ii=0;ii<numAtoms;ii++){

               //check if this input is hydrogen or oxygen based on the sampleDim ( input count)
               if(typesList[ii] == 1){
                    //this is a hydrogen atom
                    currentNet = & layers_2;
                    cout<<"USING HYDROGEN NET Input Dimension: "<< inputSizePerType[1] << " N: " <<N<<endl;
               }
               else{
                    //this is an oxygen atom
                    currentNet = & layers_1;
                    cout<<"USING OXYGEN NET Input Dimension: " << inputSizePerType[0] << " N: " <<N<<endl;
               }
               

               cout<<"Prediction " << ii << endl;
               (*currentNet).predict(input[ii], N, inputSizePerType[typesList[ii]], output);
   
               //if the finalOutput array does not exist, create and init to 0.
               if(finalOutput == nullptr){
                    cout<<endl<<"finalOutput Init"<<endl;
                    finalOutput = new T[N];
                    for(int i = 0;i < N; i++)
                         finalOutput[i] = 0;
               }

               //scores for each atom to file, sum scores to finalOutput array
               outputFile.open(filePath,ofstream::out|ofstream::app);
               outputFile<<endl<<"NEXT ATOM:"<<endl;
               for(int a = 0;a<N;a++){
                    //sum energies of all atoms for final result. 
                    finalOutput[a] += output[a];
                    outputFile<<setprecision(18)<<scientific<<output[a]<<" ";
                    if(a%3 == 2)
                         outputFile<<endl;
               }
               outputFile.close();
               
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
               if (N <= MAXSHOWRESULT){
                    cout << endl << " Final score are :" <<endl;            
                     for(int ii=0; ii<N; ii++){
                         cout << (output[ii]) << "  " ;
                    }         
               } 
               else {
                    cout << " Final score ( first " << MAXSHOWRESULT/2 << " records ):" <<endl;
                    for(int ii=0; ii<(MAXSHOWRESULT/2); ii++){
                         cout << (output[ii]) << "  " ;
                    }
                    cout << endl << " Final score ( last " << MAXSHOWRESULT/2 << " records ):" <<endl;
                    for(int ii=(N-MAXSHOWRESULT/2); ii<N; ii++){
                         cout << (output[ii]) << "  " ;
                    }                         
               }
               cout << endl;     
          }
  
          //energy conversion: 1 kcal/mol = .0433634 eV & cutoffs
           for(int a = 0;a<N;a++){
                finalOutput[a]*= (EUNIT)*cutoffs[a];
          }
          
          cout<<":::::::::::::::::::: FINAL OUTPUT::::::::::::::::: " <<endl;
          if (N <= MAXSHOWRESULT){
               cout << endl << " Final Final score are :" <<endl;            
                for(int ii=0; ii<N; ii++){
                        cout << (finalOutput[ii]) << "  " ;
               }         
          } 
          else {
               cout << " Final score ( first " << MAXSHOWRESULT/2 << " records ):" <<endl;
               for(int ii=0; ii<(MAXSHOWRESULT/2); ii++){
                    cout << (finalOutput[ii]) << "  " ;
               }
               cout << endl << " Final score ( last " << MAXSHOWRESULT/2 << " records ):" <<endl;
               for(int ii=(N-MAXSHOWRESULT/2); ii<N; ii++){
                        cout << (finalOutput[ii]) << "  " ;
               }                         
          }
          cout << endl;
          
          outputFile.open("my_y_pred.txt");
          outputFile<<" Final Output -- Summation"<<endl;
          for(int a = 0;a<N;a++){
               outputFile<<setprecision(18)<<scientific<<finalOutput[a]<<" ";
               outputFile<<endl;
          }     
          outputFile.close();
          

               
     } 
     catch (...){
          if(bias!=NULL)       delete[] bias;
          if(bias_dims!=NULL)  delete[] bias_dims;
          if(data!=NULL)       delete[] data;
          if(data_dims!=NULL)  delete[] data_dims;  
          if(output!=NULL) delete[] output;     
          if(finalOutput!=NULL) delete[] finalOutput;     
          file.close();
     }

     // Free memory of allocated arrays.
     if(bias!=NULL)       delete[] bias;
     if(bias_dims!=NULL)  delete[] bias_dims;
     if(data!=NULL)       delete[] data;
     if(data_dims!=NULL)  delete[] data_dims;     
     if(output!=NULL) delete[] output;     
     if(finalOutput!=NULL) delete[] finalOutput;  
     file.close();
     return;
}


//Instantiate all non-template class/struct/method of specific type
template struct Layer_t<double>;
template struct Layer_t<float>;

template class network_t<double>;
template class network_t<float>;


template class Layer_Net_t<double>;
template class Layer_Net_t<float>;

template void runtester<double>(const char* filename, const char* checkchar, double ** input, 
                               size_t N, double * cutoffs, const std::vector<idx_t> & typesList, 
                               const std::vector<size_t> & inputSizePerType);


template void runtester<float>(const char* filename, const char* checkchar, float ** input,
                               size_t N, float * cutoffs, const std::vector<idx_t> & typesList,
                               const std::vector<size_t> & inputSizePerType);





