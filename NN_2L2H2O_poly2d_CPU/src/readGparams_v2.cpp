#include "readGparams_v2.h"


using namespace MBbpnnPlugin;


template <typename T>
Gparams_t<T>::param_t::param_t(): nparam(0), ncol(0) {} ;


template <typename T>
Gparams_t<T>::param_t::~param_t(){} ;


// Read parameters for G-function construction
template <typename T>
Gparams_t<T>::Gparams_t(){};

template <typename T>
Gparams_t<T>::~Gparams_t(){}; 
   
// Read in params from a file; save numbers in `params`; save related atom information into `model`.
template <typename T>
int Gparams_t<T>::read_param_from_file(const char* _file){
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
               // }        
          }          
          return 0;
     } catch (const std::exception& e) {
          std::cerr << " ** Error ** : " << e.what() << std::endl;
          return 1;
     }    ;
};            


// ====================================================================
// Instanciate template with specific type
template class Gparams_t<float>;
template class Gparams_t<double>;










// nothing here but a tester
int main123(void){

     MBbpnnPlugin::Gparams_t<double> GPARAM;

     const char* file = "Gfunc_params_2Bv14.dat";

     GPARAM.read_param_from_file(file);

     // const char* file2 = "O_rad";
     // GPARAM.read_param_from_file(file2);

     // const char* file3 = "H_rad";
     // GPARAM.read_param_from_file(file3);     


     return 0;
}
