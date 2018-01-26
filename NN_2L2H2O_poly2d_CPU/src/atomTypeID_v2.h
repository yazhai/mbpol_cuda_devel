#ifndef ATOMTYPEID_H2
#define ATOMTYPEID_H2

#include <iostream>
#include <map>
#include <queue>
#include <string>
#include <vector>
#include <limits>
#include <cstring>

#include "utility.h"

typedef unsigned int idx_t;
//const idx_t DEFAULT_ID  =std::numeric_limits<idx_t>::max();
const char COMMENT_STARTER = '#' ;

// Class saving the atom type and atom ID in the model
template<typename T>
class atom_Type_ID_t2{
private:
    int load_xyz_oneblock(std::ifstream& ifs, std::vector<T>& xyz, bool if_update=false){
        std::string line;

        while(getline(ifs, line)) {
            line.erase(line.begin(), std::find_if(line.begin(), line.end(), std::bind1st(std::not_equal_to<char>(), ' ')));     
            if (line.size() > 0 && line[0] != COMMENT_STARTER)  {
                size_t natom;
                std::istringstream iss(line);
                iss >> natom;

                if(if_update) {
                    NATOM = 0;
                    TYPE_INDEX.clear();
                    TYPE_EACHATOM.clear();    
                    NATOM_ONETYPE.clear();
                    ATOMS.clear(); 
                }

                std::string comment;
                std::getline(ifs, comment);  // skip next line

                while(natom>0){
                    std::getline(ifs,line);
                    if(line.length() == 0) continue;

                    std::string element;
                    T x,y,z;

                    iss.clear();
                    iss.str(line);
                    try{
                        iss >> element >> x >> y >> z;
                    } catch(...){
                        std::ostringstream oss;
                        oss << "unexpected text  of the XYZ stream";
                        throw std::runtime_error(oss.str());
                    }

                    xyz.push_back(x);
                    xyz.push_back(y);
                    xyz.push_back(z);

                    natom --;
                    if(if_update) insert_atom(element);
                };
                break;
            };
        };
        return 0;
    };


public:

     std::vector<std::string> TYPE_INDEX;     // Type index;     
     std::vector<idx_t> TYPE_EACHATOM;        // Atom types; vector<atom_id>=type_id
     std::vector<idx_t> NATOM_ONETYPE;        // number of atoms in each type: vector<type_id> count_of_atoms
     std::vector<std::vector<idx_t>> ATOMS;   // vector<type_idx> = vector<list of atoms in this type>
     
     idx_t NATOM, NPAIR;        // number of atoms in each dimer/trimer | number of dimers/trimers
     T** XYZ;                  // xyz coordinates

     // constructor/destructor/copy-constructor     
     atom_Type_ID_t2(){
        XYZ =nullptr;
        NATOM=0;
        NPAIR=0;
     };;
     ~atom_Type_ID_t2(){
        clearMemo(XYZ);   
     };

     // query the index of a type
     idx_t get_type_idx(std::string _type, bool ifadd=false){
        auto it = find(TYPE_INDEX.begin(), TYPE_INDEX.end(), _type);
        if ( it != TYPE_INDEX.end() ) {
            // if found type, return its id;
            return (it - TYPE_INDEX.begin());
        } else {
            if(ifadd){
            // not found, create a new Type
                TYPE_INDEX.push_back(_type);
                idx_t idx = TYPE_INDEX.size() - 1;
                NATOM_ONETYPE.push_back(0);
                std::vector<idx_t> n;
                ATOMS.push_back(n);
                return idx;
            };        
        };
        return DEFAULT_ID;
    };

     // insert an atom
     idx_t insert_atom(std::string _type){
        // get atom and type index. 
        // if not found, create new label
        idx_t type_idx = get_type_idx(_type, true);

        // add to atom information
        if( type_idx != DEFAULT_ID ) {
            TYPE_EACHATOM.push_back(type_idx);
            NATOM_ONETYPE[type_idx]++;

            // atom_list of each type / do we need reorder???
            ATOMS[type_idx].push_back(NATOM++);
        };
     };
     
     // load atom xyz
     int load_xyz(const char* infile){
    
         try{             
            std::ifstream ifs(infile);
            std::vector<T> xyz_all;

            if (!ifs)
            throw std::runtime_error("could not open the XYZ file");

            load_xyz_oneblock(ifs, xyz_all, true);
            NPAIR = 0;
            while(!ifs.eof()){
                load_xyz_oneblock(ifs, xyz_all);
                NPAIR ++ ;
            }

            init_mtx_in_mem(XYZ, NPAIR, NATOM*3);
            std::memcpy(XYZ[0], &(xyz_all[0]), sizeof(T)*NATOM*NPAIR*3);
         }  catch (const std::exception& e){
                std::cerr << " ** Error ** : " << e.what() << std::endl;
                return 1;
         };
         return 0;
    };

     // load sequence 
     std::vector< std::vector<std::string> > seq;  // Sequence in getting symmetry function
     std::vector< std::vector<std::vector<idx_t> > > seq_by_idx ; // Sequence as above, but in i

    // Read in sequence from a file; save the sequnce in `seq` ready for G-fn construction
    int read_seq_from_file(const char* _file) {
        try {
            std::ifstream ifs(_file);
            std::string line;       
            
            while(getline(ifs, line)){      // get every line as in-file-stream
                
                // trimming leading space
                line.erase(line.begin(), std::find_if(line.begin(), line.end(), std::bind1st(std::not_equal_to<char>(), ' ')));      

                if ( line.size() > 0 && line[0] != COMMENT_STARTER ) {    //  if start with `COMMENT_STARTER`, it is a comment ; if length=0, it is a blank line

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
                        
                        // for every line: 
                        //        - find base atom type by first std::string
                        //        - find all other atoms types, and generate a unique matrix mapping index by the multiplication of atom type index
                        //        - save all numbers (double precision) as std::vector of vector and map it by atom type + matrix mapping index 

                        std::string seq_this_line = "";
                        idx_t seq_type = get_type_idx(record.front()) ; 
                        
                        std::vector<idx_t> seq_idx_each_line;
                        for(auto it=record.begin() ; it!=record.end(); it++){                              
                            seq_this_line += (*it); 
                            idx_t idx = get_type_idx(*it);
                            seq_idx_each_line.push_back(idx);
                        }                            
                        sort(seq_this_line.begin()+1, seq_this_line.end());      // sort out the 
                        if ( seq.size() <= seq_type ) {
                            seq.resize(seq_type+1);
                            seq_by_idx.resize(seq_type+1);
                        };                   
                        seq[seq_type].push_back(seq_this_line);                  
                        seq_by_idx[seq_type].push_back(seq_idx_each_line);
                }                                                  
            }

            return 0;
        } catch (const std::exception& e) {
            std::cerr << " ** Error ** : " << e.what() << std::endl;
            return 1;
        }
    };
     
     
     //=========================================================================
     // Following functions are only used for water dimer mbpol NN model
     //=========================================================================
     
     // loading a default setting for mbpol NN model
     void load_default_atom_id();
     void load_default_atom_seq(idx_t **& _idxary, idx_t & _size);
};

#endif
