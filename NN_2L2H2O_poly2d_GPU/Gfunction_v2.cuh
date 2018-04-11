#ifndef GFUNCTION_V2_H
#define GFUNCTION_V2_H


#include <cstdlib>
#include <vector>
#include <string>
#include <map>
#include <limits>

#include "readGparams_v2.h"
#include "atomTypeID_v2.h"
#include "timestamps.h"
#include "utility_cu.cuh"

#include<cuda.h>
#include<cublas_v2.h>


const int COL_RAD_RS = 3;
const int COL_RAD_ETA = 2; 
const int COL_RAD_TOTAL = 5;

const int COL_ANG_ETA = 2;
const int COL_ANG_ZETA = 4;
const int COL_ANG_LAMBD=3;
const int COL_ANG_TOTAL=6;

#define TILE_DIM 32

template<typename T>
__global__ void get_Gradial(T * g, T * xyz0, T * xyz1, T * p, size_t np, size_t nc, size_t N){	
	
	__shared__ T xyzTile [TILE_DIM][2];	
	int x = blockIdx.x * TILE_DIM + threadIdx.x;
	
	for(int j = 0; j < TILE_DIM; j++){
		xyzTile[threadId.x + j][0] = xyz0[( 

}



template <typename T>
class Gfunction_t{
private:

	 
//==============================================================================================
//
public:


// Variables:
atom_Type_ID_t<T> model;         // Model, save information about atom names, indexes, and types
size_t NCluster, NType;                 // Number of dimer/trimer in the model / 
idx_t* TypeStart;      // The starting atom index of one type  
idx_t* TypeNAtom;      // The count of the atoms in one type

T** xyz;                    //xyz data of atoms
T* cutoffs;                 //switching function values (Nsamples long) -- for use after NN

Gparams_t<T> gparams;                 // G-fn paramter class

std::vector<T**> G;   // G-fn matrix
std::vector < matrix_2D_d<T> *> G_d; 
std::vector<size_t> G_param_max_size;        //max size of parameters for each atom type



Gfunction_t(){
     xyz = nullptr;
     cutoffs = nullptr;
     TypeStart = nullptr;
     TypeNAtom = nullptr;
};

~Gfunction_t(){
     // clearMemo<T>(xyz);     // xyz is clean up in model member
     for(auto it=G.begin() ; it!=G.end(); it++){
          clearMemo<T>(*it);
     };
     if(TypeStart != nullptr) delete[] TypeStart;
     if(TypeNAtom != nullptr) delete[] TypeNAtom;
     if(cutoffs != nullptr) delete[] cutoffs;
};

void load_xyz(const char* file){
     model.load_xyz(file);
};

void load_paramfile(const char* file){
     if (strlen(file)>0 ){
          gparams.read_param_from_file(file);
     } else {
          load_paramfile_default();
     };
};

void load_paramfile_default(){
     gparams.read_param_from_file("H_rad");
     gparams.read_param_from_file("H_ang");
     gparams.read_param_from_file("O_rad");
     gparams.read_param_from_file("O_ang");
};


// load sequnece file
void load_seq(const char* _seqfile){
     if ( strlen(_seqfile) >0 ){
          //model2.read_seq_from_file(_seqfile);
          model.read_seq_from_file(_seqfile);
     } else {
          model.load_default_atom_seq();
     };
};

void init_G(){
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
			
			matrix_2D_d<T> * G_d_t = new matrix_2D_d<T>(model.NCLUSTER,s,G[*it]);
			G_d.push_back(G_d_t);
          }
          G_param_max_size.push_back(s);
     };
};

void load_cutoffs(){
   //fill in later
    return;
}

void make_G_XYZ(const char* _xyzFile, const char * _paramfile, const char* _ordfile){
     
     load_xyz(_xyzFile);

     load_paramfile(_paramfile);

     load_seq(_ordfile);         

     make_G();

     load_cutoffs();
}

void make_G(){


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

   	matrix_2D_d<T> * xyz_d = new matrix_2D_d<T>(model.NATOM,NCluster*3, xyz);

    	std::cout<<xyz_d->pitch<<std::endl;

   	xyz_d -> print();
	std::cout<<"Executed to here NType: "<<NType<<std::endl;
    	for(idx_t type0_id = 0; type0_id < NType; type0_id++){
		std::cout<<"Looping for typeid: "<< type0_id << std::endl;

		idx_t relation_idx = 0;    // current relation writing to this index in G
		size_t offset_by_rel = 0;  // Index already finished

    
        	for (auto rel = model.seq[type0_id].begin(); rel != model.seq[type0_id].end(); rel++ ){
			std::cout<<*rel<<std::endl;

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
                       	T** G0 = G[atom0_id] ;   // Gfn ptr

                       // second type
                       // for atom_1 in type_1;
                       // loop only atom_1>atom_0 when type_1 == type_0  
                       	for (auto atom1_id = TypeStart[type1_id]; atom1_id < TypeStart[type1_id]+TypeNAtom[type1_id] ; atom1_id++){
						// if not same type, do it normally
                                   if ( type1_id != type0_id  ){  
								std::cout<<"ok"<<std::endl;
								dim3 blockSize = dim3(32,3);
								dim3 gridSize = dim3( (NCluster + blockSize.x -1)/blockSize.x , 1); 
								get_Gradial<<< blockSize, gridSize >> (G_d[atom0_id]->dat,xyz_d->get_elem_ptr(atom1_id,0), xyz_d->get_elem_ptr(atom1_id,0), p_d, np, nc, NCluster);







							}

						 
					}
				}
			}    
			
		}
	}
/*

            idx_t type1_id = model.seq_by_idx[type_id][relation_idx][1] ; //second type in sequence

            //if radial
            if(model.seq_by_idx[type0_id][relation_idx].size() == 2){
                //pick up an atom from type_0 (first atom type in sequence)
                for(auto atom0_id = TypeStart[type0_id]; atom0_id < TypeStart[type0_id] + TypeNAtom[type0_id]; atom0_id ++){
                    

                }

            }
        }
    }

*/

/*
     // for each type (first type in seq)
     for(idx_t type_id = 0; type_id < NType; type_id ++ ){

        // iterate through all paramter seq: for "H", iterate in sequence such as "HH"-"HO"-"HHO"-"HHH" depending on the seq defined
        idx_t relation_idx = 0;    // current relation writing to this index in G
        size_t offset_by_rel = 0;  // Index already finished

        for (auto rel = model.seq[type_id].begin(); rel != model.seq[type_id].end(); rel++ ){

             size_t np = gparams.PARAMS[*rel].nparam;  // number of parameter pairs
             size_t nc = gparams.PARAMS[*rel].ncol;  // number of param in one line
             T * p = (&gparams.PARAMS[*rel].dat_h[0]); // params

             T * p_d = nullptr;                         //TODO: think of better way?
             memcpy_vec_h2d(p_d, p, np);


             idx_t type1_id = model.seq_by_idx[type_id][relation_idx][1] ; // second type in seq


             // if radial 
             if(model.seq_by_idx[type_id][relation_idx].size() == 2) { 

                   // pick up an atom for type_0 (first in seq)
                   for (auto atom0_id = TypeStart[type_id]; atom0_id < TypeStart[type_id]+TypeNAtom[type_id] ; atom0_id++){

                       // print the selected relationship
                       // std::cout << *rel << "  atom " << atom0_id << std::endl;
                       T** G0 = G[atom0_id] ;   // Gfn ptr

                       // second type
                       // for atom_1 in type_1;
                       // loop only atom_1>atom_0 when type_1 == type_0  
                       for (auto atom1_id = TypeStart[type1_id]; atom1_id < TypeStart[type1_id]+TypeNAtom[type1_id] ; atom1_id++){        

                            // if not same type, do it normally
                            if ( type1_id != type_id  )  {

                                 // print the selected atoms ! 
                                 // std::cout << atom0_id << "  " << atom1_id << std::endl;

// for (size_t batchstart = 0; batchstart < NCluster; batchstart += Cluster_Batch){
#ifdef _OPENMP
#pragma omp parallel for simd shared(atom0_id, atom1_id, batchstart, batchlimit, np, nc, G0, offset_by_rel, COL_RAD_RS, COL_RAD_ETA)
#endif  
for(size_t id_cluster = batchstart; id_cluster< batchlimit; id_cluster++)
{
                                 T d = get_dist(atom0_id, atom1_id, id_cluster);
                                 for (int ip = 0; ip < np; ip ++){

                                      // print the position in G at which the result is saved to 
                                      // size_t b = offset_by_rel;
                                      // size_t c = b + ip ;
                                      // std::cout << c << " ";
                                      
                                      G0[id_cluster][offset_by_rel + ip ]  +=  get_Gradial(d, p[ip*nc + COL_RAD_RS], p[ip*nc + COL_RAD_ETA] )  ; 
                                 }

                                 // print some nice endline
                                 // std::cout << std::endl;
}
// }
                            } else if (atom1_id > atom0_id) {

                                 // if same type, do it only when id1>id0
                                 // and assign the value to both G[id0] and G[id1] 

                                 // print the selected atoms ! 
                                 // std::cout << atom0_id << "  " << atom1_id << std::endl;
                                 T** G1 = G[atom1_id];


// for (size_t batchstart = 0; batchstart < NCluster; batchstart += Cluster_Batch){
#ifdef _OPENMP
#pragma omp parallel for simd shared(atom0_id, atom1_id, batchstart, batchlimit, np, nc, G0, G1, offset_by_rel, COL_RAD_RS, COL_RAD_ETA)
#endif
for(size_t id_cluster = batchstart; id_cluster< batchlimit; id_cluster++)
{                                        
                                 T d = get_dist(atom0_id, atom1_id, id_cluster);
                                 for (int ip = 0; ip < np; ip ++){

                                      // print the position in G at which the result is saved to 
                                      // size_t b = offset_by_rel;
                                      // size_t c = b + ip ;
                                      // std::cout << c << " ";

                                      T tmp = get_Gradial(d, p[ip*nc + COL_RAD_RS], p[ip*nc + COL_RAD_ETA] ) ;
                                      G0[id_cluster][offset_by_rel + ip ]  += tmp  ; 
                                      G1[id_cluster][offset_by_rel + ip ]  += tmp ;
                                 }

                                 // print some nice endline
                                 // std::cout << std::endl;
}
// }
                            }   // end of if type0== type1, or atom1_id>atom0_id
                       }  // end of atom1_id 

                   } // end of atom0_id  
             }   else {  // if radial -  else angular relations

                   // pick up an atom for type_0 (first in seq)
                   for (auto atom0_id = TypeStart[type_id]; atom0_id < TypeStart[type_id]+TypeNAtom[type_id] ; atom0_id++){

                       // print the selected relationship
                       // std::cout << *rel << "  atom " << atom0_id << std::endl;
                       T** G0 = G[atom0_id] ;   // Gfn ptr

                       // the third type in seq
                       idx_t type2_id = model.seq_by_idx[type_id][relation_idx][2];

                       // for atom_1 in type_1;
                       for (auto atom1_id = TypeStart[type1_id]; atom1_id < TypeStart[type1_id]+TypeNAtom[type1_id] ; atom1_id++){         

                            if (atom0_id == atom1_id) continue;

                            // if type_1 == type_2, loop only atom_1 < atom_2  
                            for (auto atom2_id = TypeStart[type2_id]; atom2_id < TypeStart[type2_id]+TypeNAtom[type2_id] ; atom2_id++){ 
                                 if (atom0_id == atom2_id) continue; 
                                 // if (atom1_id == atom2_id) continue;

                                 // if the second and third atom are not same type, do as normal
                                 if (type1_id != type2_id) {
                                 
                                      // print the selected atoms ! 
                                      // std::cout << atom0_id << "  " << atom1_id << "  " << atom2_id << std::endl;

// for (size_t batchstart = 0; batchstart < NCluster; batchstart += Cluster_Batch){
#ifdef _OPENMP
#pragma omp parallel for simd shared(atom0_id, atom1_id, atom2_id, batchstart, batchlimit, np, nc, G0, offset_by_rel, COL_ANG_ETA, COL_ANG_ZETA, COL_ANG_LAMBD)
#endif
for(size_t id_cluster = batchstart; id_cluster< batchlimit; id_cluster++)
{
                                      T dij = get_dist(atom0_id, atom1_id, id_cluster);
                                      T dik = get_dist(atom0_id, atom2_id, id_cluster);
                                      T djk = get_dist(atom1_id, atom2_id, id_cluster);

                                      for (int ip = 0; ip < np; ip ++){

                                           // print the position in G at which the result is saved to 
                                           // size_t b = offset_by_rel;
                                           // size_t c = b + ip ;
                                           // std::cout << c << " ";

                                           G0[id_cluster][offset_by_rel + ip ]  +=    get_Gangular(dij, dik, djk, p[ip*nc + COL_ANG_ETA], p[ip*nc + COL_ANG_ZETA], p[ip*nc + COL_ANG_LAMBD]) ;
                                      }

                                      // print some nice endline
                                      // std::cout << std::endl;
}
// }
                                 }  else if (atom2_id > atom1_id) {
                                      // if second and third type are the same, do once and add twice

                                      // test the selected atoms ! 
                                      // std::cout << atom0_id << "  " << atom1_id << "  " << atom2_id << std::endl;

// for (size_t batchstart = 0; batchstart < NCluster; batchstart += Cluster_Batch){
#ifdef _OPENMP
#pragma omp parallel for simd shared(atom0_id, atom1_id, atom2_id, batchstart, batchlimit, np, nc, G0, offset_by_rel, COL_ANG_ETA, COL_ANG_ZETA, COL_ANG_LAMBD)
#endif
for(size_t id_cluster = batchstart; id_cluster< batchlimit; id_cluster++)
{
                                      T dij = get_dist(atom0_id, atom1_id, id_cluster);
                                      T dik = get_dist(atom0_id, atom2_id, id_cluster);
                                      T djk = get_dist(atom1_id, atom2_id, id_cluster);

                                      for (int ip = 0; ip < np; ip ++){

                                           // print the position in G at which the result is saved to 
                                           // size_t b = offset_by_rel;
                                           // size_t c = b + ip ;
                                           // std::cout << c << " ";

                                           G0[id_cluster][offset_by_rel + ip ]  +=   get_Gangular(dij, dik, djk, p[ip*nc + COL_ANG_ETA], p[ip*nc + COL_ANG_ZETA], p[ip*nc + COL_ANG_LAMBD]) ;
                                      }

                                      // print some nice endline
                                      // std::cout << std::endl;
}
// }
                                 }  // end of if type1== type2, or atom2>atom1
                            }  // end of atom2_id
                       }  // end of atom1_id 
                  }   // end of  atom0_id
             }  // end of if_radial_else_angular 
             offset_by_rel += np;
             relation_idx++ ;
        }  // end of rel
   }  // end of type_id


}



*/
}
};
#endif
