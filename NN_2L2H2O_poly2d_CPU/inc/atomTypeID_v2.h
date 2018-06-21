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


namespace MBbpnnPlugin{ 


// Class saving the atom type and atom ID in the model
template<typename T>
class atom_Type_ID_t{
private:
    // read one cluster of atoms (6 atoms in dimer and 9 in trimer) 
    int load_xyz_oneblock(std::ifstream& ifs, std::vector<T>& xyz, bool if_update=false);

     // sort atom by type id
     // after this sort, the XYZ matrix will be organized as:
     // [dimer_0: 1st atom in type0, 2nd atom in type0, ... 1st atom in type1 ... ]
     // [dimer_1: 1st atom in type0, 2nd atom in type0, ... 1st atom in type1 ... ]
     // ...
     // atom and type information in this class is also updated accordingly.
     void sort_atom_by_type_id();

     bool ifsorted; // THIS SHOULD BE REMOVED!!!!!!!

public:

     std::vector<std::string> TYPE_INDEX;     // Type index;    
     std::vector<idx_t> TYPE_EACHATOM;        // Atom types, after sort/at read_in; vector<atom_id>=type_id
     std::vector<idx_t> NATOM_ONETYPE;        // number of atoms in each type: vector<type_id> count_of_atoms
     std::vector<std::vector<idx_t>> ATOMS;   // ATOMS in a type, after sort/at readin: vector<type_idx> = vector<list of atoms in this type>
 


     size_t NATOM, NCLUSTER, NTYPE;        // number of atoms in each dimer/trimer/cluster | number of dimers/trimers/clusters
     T** XYZ;                   // xyz coordinates
     bool XYZ_ONE_ATOM_PER_COL;            // control how XYZ is stored in memory:
                                // false: XYZ is stored NCLUSTER x (3*NATOMS) as read_in
                                // true: XYZ is transposed


     std::vector<std::vector<T> > EXTRA ;     // extra information in the xyz input file

     // sequence of atom relation
     std::vector< std::vector<std::string> > seq;  // Sequence in getting symmetry function
     std::vector< std::vector<std::vector<idx_t> > > seq_by_idx ; // Sequence as above, but using index

     // constructor/destructor/copy-constructor     
     atom_Type_ID_t();
     ~atom_Type_ID_t();

     // reset model
     void reset_xyz();
     void reset_atoms();

     // query the index of a type
     idx_t get_type_idx(std::string _type, bool ifadd=true);

     // insert an atom and return its idx
     idx_t insert_atom(std::string _type);




     // load atom xyz from file
     int load_xyz_from_file(const char* infile);

     // load atom xyz from input vectors
     int load_xyz_and_type_from_vectors(size_t nd, std::vector<T> xyz_vector, std::vector<std::string> atoms =std::vector<std::string> ()  );

     // transpose
     void transpose_xyz();
     void transpose_xyz(T** & dst);

     // Read in sequence from a file; save the sequnce in `seq` ready for G-fn construction
     int read_seq_from_file(const char* _file);
     
     
     //=========================================================================
     // Following functions are only used for water dimer mbpol NN model
     //=========================================================================
     
     // loading a default sequence for mbpol NN model
     void load_default_2h2o_3h2o_seq();
};


}; // end of namespace
#endif
