#ifndef NETWORK_H
#define NETWORK_H


#define MAXSHOWRESULT 20                // Max count of result to show for each layer output

//hdf5 file things
#define PATHTOMODEL "/model_weights"    // usual path to the group saving all the layers in HDF5 file
#define LAYERNAMES  "layer_names"       // Attribute name saving the list of layer names in HDF5
#define WEIGHTNAMES "weight_names"      // Attribute name saving the list of weight names in HDF5


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

/*Structure containing definition of Layer including: Type of layer,Weights/bias
of layer, and dimensions of layer
*/
template<typename T>
struct Layer_t{
     std::string name;
     Type_t type;
     ActType_t acttype;
      
     size_t inputs;     // number of input dimension
     size_t outputs;    // number of output dimension
     T ** weights; // weight matrix(stored as outputs x inputs)
     T * bias; // bias vector(1xoutput) or (outputx1)

     Layer_t * prev = nullptr;
     Layer_t * next = nullptr;
     
     //Default constructor
     Layer_t();

     //Dense Layer Constructor
     Layer_t(std::string _name, size_t _inputs, size_t outputs,T* _weights, T* _bias);

     //Activation Layer Constructor(using integer as type)
     Layer_t(std::string _name, int _acttype, size_t outputs);

     //Activation Layer Constructor(using enum ActType_t)
     Layer_t(std::string _name, ActType_t _acttype, size_t outputs);

     //Destructor
     ~Layer_t();

};



//Class defining movement through the Network
template<typename T>
class network_t{

public:

     //non-cblas implementation of fully Connected Forward Propogation.
     //Weights matrix and srcData are already transposed for optimal computation
     void fullyConnectedForward(const Layer_t<T> & layer,
                               const size_t N,
                               T* & srcData, T** & dstData);

     //non-cblas implementaiton of forward propogation activation function(TANH)
     void activationForward_TANH(const Layer_t<T> & layer,
                              const size_t N , 
                              T* srcData, T** & dstData);


     //non-cblas implementation of fully Connected Backward Propogation.
     //Weights matrix and srcData are already transposed for optimal computation
     void fullyConnectedBackward(const Layer_t<T> & layer,
                               const size_t N,
                               T* & srcData, T** & dstData);

     //non-cblas implementaiton of backward propogation activation function(TANH)
     void activationBackward_TANH(const Layer_t<T> & layer,
                              const size_t N , 
                              T* srcData, T** & dstData);     

          
};


// Using cblas_dgemm if cblas libraries are employed
#if defined (_USE_GSL) || defined (_USE_MKL)

template <>
void network_t<double>::fullyConnectedForward(const Layer_t<double> & layer,const size_t & N,double* & srcData, double** & dstData);

template<>
void network_t<float>::fullyConnectedForward(const Layer_t<float> & layer, const size_t & N,float* & srcData, float** & dstData);


template <>
void network_t<double>::fullyConnectedBackward(const Layer_t<double> & layer, const size_t & N,double* & srcData, double** & dstData);

template<>
void network_t<float>::fullyConnectedBackward(const Layer_t<float> & layer, const size_t & N,float* & srcData, float** & dstData);


#endif


//class for Network of Layers(Collection of Layer Structs that can implement movement of input through network)
template<typename T>
class Layer_Net_t{
private: 

     //controls propogation through network
     network_t<T> neural_net;
     
     //helper function: switch two pointers to pointers
     void switchptr(T** & alpha, T** & bravo);
     
public:
     
     //first layer(where input goes first). 
     Layer_t<T> * root = nullptr;
     
     Layer_Net_t(){};
     
     //Delete all layers, from end to root.
     ~Layer_Net_t();
     
     //inserting a dense layer
     void insert_layer(std::string &_name, size_t _inputs, size_t _outputs, 
          T * & _weights, T * & _bias);

     // Inserting an activiation layer by type (int)
     void insert_layer(std::string &_name, int _acttype, size_t _outputs);

     //Inserting an activation layer by type(enum)
     void insert_layer(std::string &_name, ActType_t _acttype, size_t _outputs);


     // Get layer ptr according to its index (start from 1 as 1st layer, 2 as second layer ...)
     Layer_t<T>* get_layer_by_seq(int _n);

     //Move through network and make prediction based on all layers.     
     void predict(T* _inputData, int _N, int _input, T* & _outputData);
};     

//Structure to hold all the different networks needed after they are built
template<typename T>
struct allNets_t{
     Layer_Net_t<T> * nets;
     size_t  numNetworks;

     //Default constructor
     allNets_t();

     //Construct from HDF5 file
     allNets_t(size_t _numNetworks, const char* filename, const char* checkchar);

     //Destructor
     ~allNets_t();
};



//Allowed Types
extern template struct Layer_t<double>;
extern template struct Layer_t<float>;

extern template struct allNets_t<double>;
extern template struct allNets_t<float>;

extern template class network_t<double>;
extern template class network_t<float>;

extern template class Layer_Net_t<double>;
extern template class Layer_Net_t<float>;


#endif
