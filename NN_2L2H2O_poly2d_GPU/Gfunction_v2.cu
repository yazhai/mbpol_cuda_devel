#include <cstdlib>
#include <vector>
#include <string>
#include <map>
#include <limits>

#include "readGparams_v2.h"
#include "atomTypeID_v2.h"
#include "timestamps.h"
#include "utility_cu.cuh"
#include "Gfunction_v2.cuh"
#include "utility.h"

#include<cuda.h>
#include<cublas_v2.h>

#define TPB 192          //threads per block
#define BPG 128           //blocks per grid


template<typename T>
Gfunction_t<T>::Gfunction_t(){
     xyz = nullptr;
     cutoffs = nullptr;
     TypeStart = nullptr;
     TypeNAtom = nullptr;
     xyz_d = nullptr;
};

template<typename T>
Gfunction_t<T>::~Gfunction_t(){
     // clearMemo<T>(xyz);     // xyz is clean up in model member
     for(auto it=G.begin() ; it!=G.end(); it++){
          clearMemo<T>(*it);
     };
     if(TypeStart != nullptr) delete[] TypeStart;
     if(TypeNAtom != nullptr) delete[] TypeNAtom;
     if(cutoffs != nullptr) cudaFree(cutoffs);
     if(xyz_d != nullptr) delete xyz_d;

};

template<typename T>
void Gfunction_t<T>::load_xyz(const char* file){
     model.load_xyz(file);
};

template<typename T>
void Gfunction_t<T>::load_paramfile(const char* file){
     if (strlen(file)>0 ){
          gparams.read_param_from_file(file);
     } else {
          load_paramfile_default();
     };
};

template<typename T>
void Gfunction_t<T>::load_paramfile_default(){
     gparams.read_param_from_file("Gfunc_params_2Bv14.dat");
     // gparams.read_param_from_file("H_rad");
     // gparams.read_param_from_file("H_ang");
     // gparams.read_param_from_file("O_rad");
     // gparams.read_param_from_file("O_ang");
};


// load sequnece file
template<typename T>
void Gfunction_t<T>::load_seq(const char* _seqfile){
     if ( strlen(_seqfile) >0 ){
          //model2.read_seq_from_file(_seqfile);
          model.read_seq_from_file(_seqfile);
     } else {
          model.load_default_atom_seq();
     };
};

template<typename T>
void Gfunction_t<T>::init_G(){
     // some preparation
     for(auto it = G.begin(); it!= G.end(); it++){
          clearMemo(*it);
     }
     G.clear();
     for(idx_t i = 0; i< model.NATOM; i++){
          T** tmp = nullptr;
          G.push_back(tmp);
     }
     NCluster = model.NCLUSTER;
     // NCluster = 1 ; // some tester
     for(idx_t type_id = 0; type_id < model.TYPE_INDEX.size(); type_id ++ ){
          // get how many params for one type
          size_t s = 0;
          for(auto it = model.seq[type_id].begin(); it!= model.seq[type_id].end() ; it++){
               s += gparams.PARAMS[*it].nparam ;
          };
          for(auto it = model.ATOMS[type_id].begin(); it!=model.ATOMS[type_id].end(); it++){
               init_mtx_in_mem(G[*it], model.NCLUSTER, s);
			
			matrix_2D_d<T> * G_d_t = new matrix_2D_d<T>(s,model.NCLUSTER);
               //G_d_t->make_transpose();
			G_d.push_back(G_d_t);
          }
          G_param_max_size.push_back(s);
     };
};

template<typename T>
void Gfunction_t<T>::load_cutoffs(){
    T ri = 4.5, rf = 6.5;
    if(cutoffs != nullptr)  clearMemo_d(cutoffs);
    init_vec_in_mem_d(cutoffs , NCluster);

    if(TypeNAtom[0] == 2){   
        fswitch_2b<T><<<BPG,TPB>>>(cutoffs, xyz_d->get_elem_ptr(0,0), xyz_d->get_elem_ptr(1,0),ri,rf, NCluster);
    }
    else{
        fswitch_3b<T><<<BPG,TPB>>>(cutoffs, xyz_d->get_elem_ptr(0,0), xyz_d->get_elem_ptr(1,0), xyz_d->get_elem_ptr(2,0),0.0,ri, NCluster);  
    }

    // std::cerr<<"INSIDE OF LOAD CUTOFFS FUNCTION , DEVICE VECTOR IS: "<<std::endl;
    // printDeviceVector(cutoffs,NCluster);
    return; 
}

template<typename T>
void Gfunction_t<T>::scale_G(const char ** _scaleFiles){

     // //currently assumes that same number of scaling files provided as different atom types, and they are in the same order
     T ** scales[NType];
     T * scales_d [NType];
     size_t pitch[NType];
     size_t rows[NType];
     size_t col; //always 1
   
     for(int i = 0; i<NType; i++){
          scales[i] = nullptr;
          scales_d[i] = nullptr;
          read2DArrayfile<T>(scales[i], rows[i], col, _scaleFiles[i]);
          memcpy_vec_h2d(scales_d[i], *(scales[i]), rows[i]);
          
      }
     

     for(int i = 0; i< model.NATOM; i++){
          size_t paramSize = G_param_max_size[model.TYPE_EACHATOM[i]];
          scale <<< BPG, TPB >>> (NCluster, paramSize, scales_d[model.TYPE_EACHATOM[i]], G_d[i]->dat, G_d[i]->pitch);
     } 



     // std::string filename = "O_out_scale.dat";
     // G_d[0]->printFile(filename.c_str());
     // std::cout<<std::endl;


}



template<typename T>
void Gfunction_t<T>::make_G_XYZ(const char* _xyzFile, const char * _paramfile, const char* _ordfile, const char** _scaleFiles){
     
     cublas_start();

     load_xyz(_xyzFile);

     load_paramfile(_paramfile);

     load_seq(_ordfile);         

     make_G();

     scale_G(_scaleFiles);

     load_cutoffs();

     cublas_end();
}

template<typename T>
void Gfunction_t<T>::make_G(){
     timers.insert_random_timer( id, 0, "Gfn_total");
     timers.timer_start(id);

     model.sort_atom_by_type_id(); // sort atoms according to type  
     init_G(); 

     
     NType = model.TYPE_INDEX.size();
     TypeStart = new idx_t[NType];
     TypeNAtom = new idx_t[NType];


     {
          idx_t idx = 0;
          idx_t count = 0;
          for(auto it = model.NATOM_ONETYPE.begin(); it!= model.NATOM_ONETYPE.end(); it++ ){
               TypeNAtom[idx] = *it ; 
               TypeStart[idx] = count ;
               count += *it;
               idx ++ ;
          }
     }
          
     model.transpose_xyz();

     T ** xyz = model.XYZ;

     if(xyz_d != nullptr) 
          delete xyz_d;
          
     xyz_d = new matrix_2D_d<T>(model.NATOM,NCluster*3, xyz);

          for(idx_t type0_id = 0; type0_id < NType; type0_id++){

          idx_t relation_idx = 0;    // current relation writing to this index in G
          size_t offset_by_rel = 0;  // Index already finished

     
               for (auto rel = model.seq[type0_id].begin(); rel != model.seq[type0_id].end(); rel++ ){
                    size_t np = gparams.PARAMS[*rel].nparam;  // number of parameter pairs

                    size_t nc = gparams.PARAMS[*rel].ncol;  // number of param in one line

                    T * p = (&gparams.PARAMS[*rel].dat_h[0]); // params	
                    T * p_d = gparams.PARAMS[*rel].dat_d;

                    idx_t type1_id = model.seq_by_idx[type0_id][relation_idx][1] ; // second type in seq

               // if radial 
                    if(model.seq_by_idx[type0_id][relation_idx].size() == 2) { 

                    // pick up an atom for type_0 (first in seq)
                    for (auto atom0_id = TypeStart[type0_id]; atom0_id < TypeStart[type0_id]+TypeNAtom[type0_id] ; atom0_id++){

                         // print the selected relationship
                         // std::cout << *rel << "  atom " << atom0_id << std::endl;

                         // second type
                         // for atom_1 in type_1;
                         // loop only atom_1>atom_0 when type_1 == type_0  
                              for (auto atom1_id = TypeStart[type1_id]; atom1_id < TypeStart[type1_id]+TypeNAtom[type1_id] ; atom1_id++){
                              // if not same type, do it normally
                              if ( type1_id != type0_id   ){  

                                   get_Gradial<<< BPG,TPB,(np*nc)*sizeof(T)>>> (G_d[atom0_id]->dat,G_d[atom0_id]->pitch, xyz_d->get_elem_ptr(atom0_id,0), 
                                        xyz_d->get_elem_ptr(atom1_id,0), p_d, np, nc, NCluster, offset_by_rel);

                              }
                              else if (atom1_id > atom0_id) {

                                   get_Gradial2<<< BPG, TPB, (np*nc)*sizeof(T) >>>(G_d[atom0_id]->dat, G_d[atom1_id]->dat, G_d[atom0_id]->pitch,
                                        xyz_d->get_elem_ptr(atom0_id,0), xyz_d->get_elem_ptr(atom1_id,0), p_d, np, nc, NCluster, offset_by_rel);
                              }
                                   
                         }
                    }
               }
               //angular
               else{  
               for (auto atom0_id = TypeStart[type0_id]; atom0_id < TypeStart[type0_id]+TypeNAtom[type0_id] ; atom0_id++){
                    // the third type in seq
                    idx_t type2_id = model.seq_by_idx[type0_id][relation_idx][2];
                    // for atom_1 in type_1;
                    for (auto atom1_id = TypeStart[type1_id]; atom1_id < TypeStart[type1_id]+TypeNAtom[type1_id] ; atom1_id++){         

                         if (atom0_id == atom1_id) continue;

                         // if type_1 == type_2, loop only atom_1 < atom_2  
                         for (auto atom2_id = TypeStart[type2_id]; atom2_id < TypeStart[type2_id]+TypeNAtom[type2_id] ; atom2_id++){ 
                              if (atom0_id == atom2_id) continue; 
                              // if (atom1_id == atom2_id) continue;

                              // if the second and third atom are not same type, do as normal
                              if (type1_id != type2_id || atom2_id>atom1_id) {

                                   get_Gangular<<< BPG, TPB, (np*nc)*sizeof(T)>>> (G_d[atom0_id]->dat, G_d[atom0_id]->pitch, xyz_d->get_elem_ptr(atom0_id,0), 
                                        xyz_d->get_elem_ptr(atom1_id,0), xyz_d->get_elem_ptr(atom2_id,0), p_d, np, nc, NCluster, offset_by_rel);
                              }
                              // }  else if (atom2_id > atom1_id) {

                              //     get_Gangular<<< BPG, TPB >>> (G_d[atom0_id]->dat,G_d[atom0_id]->pitch, xyz_d->get_elem_ptr(atom0_id,0), 
                              //     xyz_d->get_elem_ptr(atom1_id,0), xyz_d->get_elem_ptr(atom2_id,0), p_d, np, nc, NCluster, offset_by_rel); 
                              // }
                         }
                    }

               }
               }
               offset_by_rel += np;
               relation_idx ++; 
               
          }
     }
     timers.timer_end(id);
     timers.get_all_timers_info();
     timers.get_time_collections();


     std::cout<<"End of Gfn"<<std::endl;
}

template class Gfunction_t<double>;
template class Gfunction_t<float>;
