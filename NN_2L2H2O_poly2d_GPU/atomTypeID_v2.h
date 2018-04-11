#ifndef ATOMTYPEID_H
#define ATOMTYPEID_H

#include <iostream>
#include <map>
#include <queue>
#include <string>
#include <vector>
#include <limits>
#include <cstring>
#include <algorithm>
#include <iomanip>

#include "utility.h"


const idx_t DEFAULT_ID  =std::numeric_limits<idx_t>::max();
const char COMMENT_STARTER = '#' ;

// Class saving the atom type and atom ID in the model
template<typename T>
class atom_Type_ID_t{
private:
    // read one cluster of atoms (6 atoms in dimer and 9 in trimer) 
    int load_xyz_oneblock(std::ifstream& ifs, std::vector<T>& xyz, bool if_update=false){
        std::string line;
        while(getline(ifs, line)) {
            line.erase(line.begin(), std::find_if(line.begin(), line.end(), std::bind1st(std::not_equal_to<char>(), ' ')));     
            if (line.size() > 0 && line[0] != COMMENT_STARTER)  {
                // First line is natom
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

                // second line is some extra info
                {
                    std::string comment;
                    std::getline(ifs, comment);         // skip next line
                    std::istringstream iss(comment);    // get every line as in-string-stream 
                    std::vector<std::string> record;
                    
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
                    //        - save all numbers (double/single precision) as vector of vector  
                    for(auto it=record.begin(); it!=record.end(); it++){                              
                         if ( IsFloat<T>(*it)){
                              T f = return_a_number<T>(*it);               
                              currnumbers.push_back(f);               
                         } 
                    };
                    EXTRA.push_back(currnumbers);
                }

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
                return 0;
            };
        };
        return 1;
    };


public:

     std::vector<std::string> TYPE_INDEX;     // Type index;     
     std::vector<idx_t> TYPE_EACHATOM, TYPE_EACHATOM_READIN;        // Atom types, after sort/at read_in; vector<atom_id>=type_id
     std::vector<idx_t> NATOM_ONETYPE;        // number of atoms in each type: vector<type_id> count_of_atoms
     std::vector<std::vector<idx_t>> ATOMS, ATOMS_READIN;   // ATOMS in a type, after sort/at readin: vector<type_idx> = vector<list of atoms in this type>
 


     idx_t NATOM, NCLUSTER;        // number of atoms in each dimer/trimer/cluster | number of dimers/trimers/clusters
     T** XYZ;                   // xyz coordinates
     bool XYZ_TRANS;            // control how XYZ is stored in memory:
                                // false: XYZ is stored as read_in []
                                // true: XYZ is transposed
     bool ifsorted;             // if the XYZ has been sorted.
     std::vector<std::vector<T> > EXTRA ;     // extra information in the xyz input file

     // sequence of atom relation
     std::vector< std::vector<std::string> > seq;  // Sequence in getting symmetry function
     std::vector< std::vector<std::vector<idx_t> > > seq_by_idx ; // Sequence as above, but using index

     // constructor/destructor/copy-constructor     
     atom_Type_ID_t(){
        XYZ =nullptr;
        NATOM=0;
        NCLUSTER=0;
        ifsorted = false;
        XYZ_TRANS= false;
     };
     ~atom_Type_ID_t(){
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
                ATOMS_READIN.push_back(n);
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
            TYPE_EACHATOM_READIN.push_back(type_idx);
            NATOM_ONETYPE[type_idx]++;

            // atom_list of each type / do we need reorder???
            ATOMS_READIN[type_idx].push_back(NATOM++);
        };
     };
     
     // sort atom by type id
     // after this sort, the XYZ matrix will be organized as:
     // [dimer_0: 1st atom in type0, 2nd atom in type0, ... 1st atom in type1 ... ]
     // [dimer_1: 1st atom in type0, 2nd atom in type0, ... 1st atom in type1 ... ]
     // ...
     // atom and type information in this class is also updated accordingly.

     void sort_atom_by_type_id(){         
         
          if (ifsorted) return ;

         bool if_do_a_trans = false;
         if(!XYZ_TRANS) {
             transpose_xyz();
             if_do_a_trans = true;
         };
         
         T** xyz_old = XYZ;
         XYZ = nullptr;
         init_mtx_in_mem(XYZ, NATOM*3, NCLUSTER);
         int start_row_index = 0;
         

         std::vector<std::vector<idx_t>> atoms_tmp;
         std::vector<idx_t> type_eachatom_tmp;
         idx_t atom_id=0;
         idx_t type_id=0;
         for (auto it = ATOMS_READIN.begin(); it!= ATOMS_READIN.end(); it++){
             std::vector<idx_t> atoms_onetype;
            for (auto it2 = (*it).begin(); it2 != (*it).end(); it2++){
                std::memcpy(XYZ[start_row_index], xyz_old[3*(*it2)], sizeof(T)*3*NCLUSTER);
                start_row_index += 3;
                atoms_onetype.push_back(atom_id++);
                type_eachatom_tmp.push_back(type_id);
            } ;
            type_id++;
            atoms_tmp.push_back(atoms_onetype);
         };
         ATOMS = atoms_tmp;
         TYPE_EACHATOM = type_eachatom_tmp;
        clearMemo(xyz_old);

         if(if_do_a_trans) {
             transpose_xyz();
         };

         ifsorted = true;

     };



     // load atom xyz
     int load_xyz(const char* infile){
         try{             
            std::ifstream ifs(infile);
            std::vector<T> xyz_all;

            if (!ifs)
            throw std::runtime_error("could not open the XYZ file");

            load_xyz_oneblock(ifs, xyz_all, true);
            NCLUSTER = 1;
            while(!ifs.eof()){
                if ( load_xyz_oneblock(ifs, xyz_all) == 0)   NCLUSTER ++ ;
            }
            
            //NCLUSTER --;        //myedit - for some reason nclusters is one too big after processing, however xyz is correct
            init_mtx_in_mem(XYZ, NCLUSTER, NATOM*3);

            std::memcpy(XYZ[0], &(xyz_all[0]), sizeof(T)*NATOM*NCLUSTER*3);

            XYZ_TRANS = false ;
            ifsorted = false;
            ATOMS = ATOMS_READIN;
            TYPE_EACHATOM = TYPE_EACHATOM_READIN;
         }  catch (const std::exception& e){
                std::cerr << " ** Error ** : " << e.what() << std::endl;
                return 1;
         };
         return 0;
    };

    void transpose_xyz(){
        T ** xyz_tmp = nullptr;

        if (XYZ_TRANS) {
            init_mtx_in_mem(xyz_tmp, NATOM*3, NCLUSTER);
            std::memcpy(xyz_tmp[0], XYZ[0], sizeof(T)*NCLUSTER*NATOM*3);
            clearMemo(XYZ);
            init_mtx_in_mem(XYZ, NCLUSTER, NATOM*3);
            transpose_mtx(XYZ, xyz_tmp, NATOM*3, NCLUSTER); 
        } else {
            init_mtx_in_mem(xyz_tmp, NCLUSTER, NATOM*3);
            std::memcpy(xyz_tmp[0], XYZ[0], sizeof(T)*NCLUSTER*NATOM*3);
            clearMemo(XYZ);
            init_mtx_in_mem(XYZ, NATOM*3, NCLUSTER);
            transpose_mtx(XYZ, xyz_tmp, NCLUSTER, NATOM*3);        
        } ;
        XYZ_TRANS = (!XYZ_TRANS);
        clearMemo(xyz_tmp);
    }


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
     
     // loading a default sequence for mbpol NN model
     void load_default_atom_seq(){
         read_seq_from_file("Gfn_order.dat");
     };
};

#endif
