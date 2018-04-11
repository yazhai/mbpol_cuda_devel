#ifndef GPARAMS_CUH
#define GPARAMS_CUH

#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <iomanip>

// #include "atomTypeID_v2.h"
#include "utility.h"
#include "utility_cu.cuh"

//#ifndef CONST_VAR_INSTANT 
//const int SYMBOL_COUNT = 4096;
//extern __constant__ double PARAMS[SYMBOL_COUNT] ; // constant memory in CUDA. Reserve 32K.
//#endif

// Read parameters for G-function construction
template <typename T>
struct Gparams_t {
public:

     struct param_t {
          std::vector< T > dat_h;
          idx_t nparam, ncol;
		T * dat_d;           
          param_t() : nparam(0), ncol(0) {};
          ~param_t(){};                 
     };
     
     
     
     // member variables
     std::map< std::string, param_t> PARAMS;  // save read-in parameters
     
     
     // constructor/destructor
     Gparams_t(){
	};
     ~Gparams_t(){
		for(auto it = PARAMS.begin(); it != PARAMS.end(); it++){

			if(it->second.dat_d != nullptr)
				delete it->second.dat_d;
		}
	}; 
   
     // Read in params from a file; save numbers in `params`; save related atom information into `model`.
     int read_param_from_file(const char* _file){
         try {
               std::ifstream ifs(_file);
               std::string line;       
                        
               while(getline(ifs, line)){      // get every line as in-file-stream
                    std::istringstream iss(line);   // get every line as in-string-stream 
                    std::vector<std::string> record;
                    
                    // split the records in line by space
                    while(iss){
                         std::string next;
                         if (!getline(iss,next, ' ') ) break;
                         if (next != ""){
                              record.push_back(next);
                         }
                    }
                    
                    std::vector<T> currnumbers;   // saved numbers in current line
                    std::string atom_type = "";   // the atom types in the line 

                    // for every line: 
                    //        - find all atoms types, and save the types into one string
                    //        - save all numbers (double/single precision) as a vector and map it by atom type + matrix mapping index 
                    for(auto it=record.begin(); it!=record.end(); it++){                              
                         if ( IsFloat<T>(*it)){
                              T f = return_a_number<T>(*it);               
                              currnumbers.push_back(f);               
                         } else {                        
                              atom_type += (*it) ;           
                         }
                    };
                    sort(atom_type.begin()+1, atom_type.end());  // sort the sequnce by alphabet. Make the char consistent at comparison.

                    // a little cleaning up for unuseful numbers
                    // this depends on the input style
                    // update: this part is not a general style, so it is removed or implemented in another part of the code
                    // if ( currnumbers.size() >0 ){   
                    //      currnumbers.erase(currnumbers.begin(), currnumbers.begin()+2);   // remove the first 2 useless numbers 
                    //      currnumbers.erase(currnumbers.end()-1, currnumbers.end()); // remove the last number
                         
                    PARAMS[atom_type].dat_h.insert(PARAMS[atom_type].dat_h.end(), currnumbers.begin(), currnumbers.end() );            
                    PARAMS[atom_type].nparam ++ ;
                    PARAMS[atom_type].ncol = currnumbers.size();
				PARAMS[atom_type].dat_d = nullptr;

					
                    // }        
               }

			for(auto it = PARAMS.begin(); it!= PARAMS.end(); it++){
				std::cout<<"trying... " << it -> second.dat_h[0]<<" "<< it->second.nparam<< " ";
				memcpy_vec_h2d(it->second.dat_d, &(it->second.dat_h[0]), it->second.nparam * it->second.ncol);
				
			}
				          
               return 0;
         } catch (const std::exception& e) {
             std::cerr << " ** Error ** : " << e.what() << std::endl;
             return 1;
         }    ;
     };            
} ;


#endif
