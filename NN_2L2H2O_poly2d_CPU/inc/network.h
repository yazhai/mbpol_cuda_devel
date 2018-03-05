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
	Layer_t(std::string _name, int _acttype);

	//Activation Layer Constructor(using enum ActType_t)
	Layer_t(std::string _name, ActType_t _acttype);

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
                          	size_t & input, size_t & output, size_t & N,
                          	T* & srcData, T** & dstData);

	//non-cblas implementaiton of forward propogation activation function(TANH)
	void activationForward_TANH(const int & output,const int & N , T** srcData, T** & dstData);

		
};


// Using cblas_dgemm if cblas libraries are employed
#if defined (_USE_GSL) || defined (_USE_MKL)
template <>
void network_t<double>::fullyConnectedForward(const Layer_t<double> & layer,size_t & input, size_t & output, size_t & N,double* & srcData, double** & dstData);

template<>
void network_t<float>::fullyConnectedForward(const Layer_t<float> & layer,size_t & input, size_t & output, size_t & N,float* & srcData, float** & dstData);
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
	void insert_layer(std::string &_name, int _acttype);

	//Inserting an activation layer by type(enum)
	void insert_layer(std::string &_name, ActType_t _acttype);


	// Get layer ptr according to its index (start from 1 as 1st layer, 2 as second layer ...)
	Layer_t<T>* get_layer_by_seq(int _n);

	//Move through network and make prediction based on all layers.	
	void predict(T* _inputData, int _N, int _input, T* & _outputData, unsigned long int& _outsize);
};	



/* TESTER FUNCTION 
	Input:	filename -- weights and bias datafile -- defined in "fullTester.cpp"
			checkchar - character used in processing datafile to differentiate between weights and biases -- defined in "fullTester.cpp"
			input -- 2-d array of Gfn outputs (numAtoms x (N*sampleDim[i]))
			numAtoms -- number of atoms to be proccessed (first dimension of input array)
			sampleCount -- N, the number of samples for each atom (second dimension of input array)
			sampleDim -- a 1 x numAtoms sized array containing information for the number of inputs per sample
	Result:
			Printing of first/last 10 scores for each atom
			Printing of first/last 10 scores for the final output(summation of all atoms)
			Store all scores of each atom in file -- "NN_final.out"
			Store final output score(summation of all atoms scores) in file -- "my_y_pred.txt"
			The above file can be compared with file "y_pred.txt" which contains outputs from python implementation
*/
template <typename T>
void runtester(const char* filename, const char* checkchar, T** input, size_t numAtoms, size_t sampleCount, size_t * sampleDim, T * cutoffs);


//Allowed Types
extern template struct Layer_t<double>;
extern template struct Layer_t<float>;

extern template class network_t<double>;
extern template class network_t<float>;

extern template class Layer_Net_t<double>;
extern template class Layer_Net_t<float>;

extern template void runtester<double>(const char* filename, const char* checkchar, double ** input, size_t numAtoms, 
						size_t sampleCount, size_t * sampleDim, double * cutoffs);


extern template void runtester<float>(const char* filename, const char* checkchar, float ** input, size_t numAtoms, 
						size_t sampleCount, size_t * sampleDim, float * cutoffs);


#endif
