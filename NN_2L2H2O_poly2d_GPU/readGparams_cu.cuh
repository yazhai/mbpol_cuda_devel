#ifndef GPARAMS_CUH
#define GPARAMS_CUH

#include <vector>
#include <string>
#include <map>
#include <algorithm>

#include "atomTypeID.h"
#include "utility_cu.cuh"


// The comment line in "seq.file" starts with following symbol
#define COMMENT_IN_SEQ '#'

// Read parameters for G-function construction
template <typename T>
struct Gparams_t {
public:

     struct param_t {
          T* dat ;
          size_t nparam, ncol;          
               
          param_t() : nparam(0), ncol(0), dat(nullptr){};
          param_t(T* _dat, size_t _nparam, size_t _ncol) : dat(_dat), nparam(_nparam), ncol(_ncol) {}; 
          ~param_t(){
               clearMemo_d(dat);          
          };
          
          T* get_param_dptr( size_t param_idx, size_t col_idx){
               return &( dat[param_idx*ncol+col_idx] ) ;                              
          };                        
     };
     
     
     
     // member variables
     std::map<std::string,  std::map<idx_t, std::vector<std::vector<T> > > > params;  
     std::map<std::string,  std::map<idx_t, param_t*> > params_d;
               
     std::map<std::string,  std::vector<idx_t> > seq;                   
     
     
     // constructor/destructor
     Gparams_t(){ 
          clearParams_d();
     };
     ~Gparams_t(){
          clearParams_d();
     }; 
     //copy-structor is missing here.
     
     // Memory clearning in devices
     void clearParams_d(){
          for(auto it = params_d.begin(); it!= params_d.end() ; it++){
               for(auto it2 = it->second.begin(); it2 != it->second.end(); it2++){
                    delete it2->second;
               };
          }; 
     };


     // Stor Params from Host to Device
     void updateParam_d(){
          clearParams_d();          
          for(auto it = params.begin(); it != params.end(); it++){
               for(auto it2 = it->second.begin() ; it2 != it->second.end(); it2++){               
                    size_t m = it2->second.size();
                    size_t n = (m>0) ? it2->second[0].size() : 0;
                    T* tmp_ptr = nullptr;
                    init_vec_in_mem_d(tmp_ptr, m*n);           
                    for(int ii = 0 ; ii < m ; ii ++ ){
                         checkCudaErrors( cudaMemcpy( &tmp_ptr[ii*n], &(it2->second[ii]), sizeof(T)*n, cudaMemcpyHostToDevice) );                    
                    }                    
                    param_t* tmp_param = new param_t(tmp_ptr, m, n) ;               
                    params_d[it->first][it2->first] = tmp_param;  // params_d[atom_type][atom_relation] -> a param_t object        SOMETHING TO PLAY WITH FOR PERFORMANCE                    
               }
          }
     }
     
     // Read in params from a file; save numbers in `params`; save related atom information into `model`.
     int read_param_from_file(const char* p, atom_Type_ID_t & model){
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



     // Read in sequence from a file; save the sequnce in `seq` ready for G-fn construction
     int read_seq_from_file(const char* _file, atom_Type_ID_t & model) {
          try {
               std::ifstream ifs(_file);
               std::string line;       
               
               
               while(getline(ifs, line)){      // get every line as in-file-stream
                    
                    // trimming leading space
                    line.erase(line.begin(), std::find_if(line.begin(), line.end(), std::bind1st(std::not_equal_to<char>(), ' ')));      
                    
                             
                    if ( line.size() > 0 && line[0] != COMMENT_IN_SEQ ) {    //  if start with `COMMENT_IN_SEQ`, it is a comment ; if length=0, it is a blank line

                         std::stringstream iss(line);   // get the line as in-string-stream 
                         std::vector<std::string> record;
                    
                         // split the records in line by space
                         while(iss){
                              std::string next;
                              if (!getline(iss,next, ' ') ) break;
                              if (next != ""){
                                   record.push_back(next);
                              }                  
                         }
                    
                         
                         std::string atom_type = "";        // the first atom in the line is the main atom 
                         idx_t atom_relation =1;       // the relationship index ( = multiple of atom type index other than the main atom )

                         // for every line: 
                         //        - find base atom type by first string
                         //        - find all other atoms types, and generate a unique matrix mapping index by the multiplication of atom type index
                         //        - save all numbers (double precision) as vector of vector and map it by atom type + matrix mapping index 
                         for(auto it=record.begin(); it!=record.end(); it++){                              
                                      
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
                         seq[atom_type].push_back(atom_relation);
                    }                                                  
               }          
               return 0;
          } catch (const std::exception& e) {
             std::cerr << " ** Error ** : " << e.what() << std::endl;
             return 1;
         }
     }

     // Make the sequence in a weired order
     void make_seq_default(){     
          for(auto it=params.begin(); it!=params.end(); it++){          
               for(auto it2=it->second.begin(); it2!=it->second.end(); it2++){
                    seq[it->first].push_back(it2->first);           
               }
          }
     }     
     
     
private:
     T return_a_number(std::string _string);   
     
     
     
       
} ;


#endif
