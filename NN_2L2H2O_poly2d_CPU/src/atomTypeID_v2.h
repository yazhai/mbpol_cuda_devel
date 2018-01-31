#ifndef ATOMTYPEID_H2
#define ATOMTYPEID_H2




//const idx_t DEFAULT_ID  =std::numeric_limits<idx_t>::max();
const char COMMENT_STARTER = '#' ;

// Class saving the atom type and atom ID in the model
template<typename T>
class atom_Type_ID_t2{
private:
    typedef unsigned int idx_t;
    int load_xyz_oneblock(std::ifstream& ifs, std::vector<T>& xyz, bool if_update=false);
public:
     std::vector<std::string> TYPE_INDEX;     // Type index;     
     std::vector<idx_t> TYPE_EACHATOM;        // Atom types; vector<atom_id>=type_id
     std::vector<idx_t> NATOM_ONETYPE;        // number of atoms in each type: vector<type_id> count_of_atoms
     std::vector<std::vector<idx_t>> ATOMS;   // vector<type_idx> = vector<list of atoms in this type>
     
     idx_t NATOM, NPAIR;        // number of atoms in each dimer/trimer | number of dimers/trimers
     T** XYZ;                  // xyz coordinates

     // constructor/destructor/copy-constructor     
    atom_Type_ID_t2();
    ~atom_Type_ID_t2();

    idx_t get_type_idx(std::string _type, bool ifadd=false);
    idx_t insert_atom(std::string _type);
    int load_xyz(const char* infile);

    // load sequence 
     std::vector< std::vector<std::string> > seq;  // Sequence in getting symmetry function
     std::vector< std::vector<std::vector<idx_t> > > seq_by_idx ; // Sequence as above, but in i

    // Read in sequence from a file; save the sequnce in `seq` ready for G-fn construction
    int read_seq_from_file(const char* _file);

     //=========================================================================
     // Following functions are only used for water dimer mbpol NN model
     //=========================================================================
     
     // loading a default setting for mbpol NN model
     void load_default_atom_id();
     void load_default_atom_seq(idx_t **& _idxary, idx_t & _size);

    
};

#endif
