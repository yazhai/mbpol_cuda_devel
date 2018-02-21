#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>
#include <memory>
#include <cstdlib>
#include <algorithm>

#include "readGparams.h"
#include "atomTypeID.h"
#include "utility.h"




using namespace std;


template<>
double Gparams_t<double>::return_a_number(string _string){
     return  stod ( _string );     
}

template<>
float Gparams_t<float>::return_a_number(string _string){
     return  stof ( _string );     
}



// Read in params from a file; save numbers in `params`; save related atom information into `model`.
template<typename T>
int Gparams_t<T>::read_param_from_file(const char* p, atom_Type_ID_t & model){
    try {
        std::ifstream ifs(p);
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
            std::string atom_type = "";        // the first atom in the line is the main atom 
            idx_t atom_relation =1;       // the relationship index ( = multiple of atom type index other than the main atom )
            
            // for every line: 
            //        - find param's atom type by first string
            //        - find all other atoms types, and generate a unique matrix mapping index by the multiplication of atom type index
            //        - save all numbers (double/single precision) as vector of vector and map it by atom type + matrix mapping index 
            for(auto it=record.begin(); it!=record.end(); it++){                              
                    if ( IsFloat<T>(*it)){
                        T f = return_a_number(*it);               
                        currnumbers.push_back(f);               
                    } else {                        
                        if(atom_type == "")  atom_type = *it;
                        
                        auto it2 = model.types.find(*it) ;                          
                        idx_t curridx =1; 
                        
                        if( it2 == model.types.end() ){  
                            curridx = model.insert_type(*it); 
                        }else {
                            curridx = model.types[*it]->id;
                        }
                        
                        atom_relation *= curridx;               
                    }
            }
                                                                                                                                                    
            if ( currnumbers.size() >0 && atom_relation>1 ){   
                    params[atom_type][atom_relation].push_back(currnumbers);            
            }        
        }          
        return 0;
    } catch (const std::exception& e) {
        std::cerr << " ** Error ** : " << e.what() << std::endl;
        return 1;
    }    ;
};

template int Gparams_t<double>::read_param_from_file(const char* p, atom_Type_ID_t & model);
template int Gparams_t<float>::read_param_from_file(const char *p, atom_Type_ID_t & model);