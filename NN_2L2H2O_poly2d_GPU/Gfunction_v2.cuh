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

//#define TILE_DIM 24 // 3 (x,y,z triplet) * 8
//#define BLOCK_ROWS = 8
#define TPB 64
#define BPG 8
#define MAXPARAM 50



// Get cosine
template<typename T>
__device__ T get_cos_d(T Rij_d, T Rik_d, T Rjk_d) {
     //cosine of the angle between two vectors ij and ik    
     T Rijxik = Rij_d*Rik_d;    
     if ( Rijxik != 0 ) {
          return  ( ( Rij_d*Rij_d + Rik_d*Rik_d - Rjk_d*Rjk_d )/ (2.0 * Rijxik) );
     } else {
          //rst_d =  std::numeric_limits<T>::infinity();
          return  -98765.4321;   // some strange value
     }
};


template<typename T>
__device__ T get_dist2(const T * d){
     T a = d[0] - d[1];
     T b = d[2] - d[3];
     T c = d[4] - d[5];
     return sqrt(a*a + b*b + c*c);
}

template<typename T>
__device__ T get_dist3(const T * d, size_t i1, size_t i2){
     T a = d[i1] - d[i2];
     T b = d[3+i1] - d[3+i2];
     T c = d[6+i1] - d[6 + i2];
     return sqrt(a*a + b*b + c*c);
}

// Get G_radial
template<typename T>
__device__ T get_Gradial_d(T Rij_d, T Rs, T eta){
     T G_rad = exp( -eta * ( (Rij_d-Rs)*(Rij_d-Rs) )  )  ;
     return G_rad;
};

// Get G_angular
template<typename T>
__device__ T get_Gangular_d(T Rij_d, T Rik_d, T Rjk_d, T eta, T zeta, T lambd){         
     T  G_ang =   2 * pow( (1.0 + lambd* get_cos_d(Rij_d, Rik_d, Rjk_d))/2.0, zeta) 
                         * exp(-eta*  ( (Rij_d+Rik_d+Rjk_d)*(Rij_d+Rik_d+Rjk_d) ) );    
     
     return G_ang ;    
};

template<typename T>
__global__ void get_Gradial(T * g, size_t pitch, T * xyz0, T * xyz1, T * p, size_t np, size_t nc, size_t N, size_t offset){	
	
     __shared__ T xyzShare [TPB*6];
     // __shared__ T pShare [500];
     __shared__ T results [TPB*MAXPARAM];

     int tid = threadIdx.x + blockDim.x * blockIdx.x;
     int stride = blockDim.x * gridDim.x;



     while(tid < N){

          //load params 
          // int x = np*nc/TPB + 1;    //x is scaling factor for parameter loading, needs to changed based on numbers of threads used
          // for(int i = 0; i<x; i++){
          //      if((threadIdx.x*x+i)<np*nc)
          //           pShare[threadIdx.x*x+i] = p[threadIdx.x*x+i];     
          // }
    
          for(int i = 0; i<3; i++){
               xyzShare[threadIdx.x*6+i*2] = xyz0[N*i+tid];
          }

          for(int j = 0; j<3; j++){
               xyzShare[threadIdx.x*6 + (j*2)+1] = xyz1[N*j +tid];
          }


          __syncthreads();

          T distance = get_dist2<T> (& xyzShare[threadIdx.x*6]);

          for (int ip = 0; ip < np; ip ++){

               //id_cluster = tid
               //offset_by_rel must be passed
             //  g[tid*pitch/sizeof(T) + offset + ip] += get_Gradial_d(distance, pShare[ip*nc + COL_RAD_RS], pShare[ip*nc + COL_RAD_ETA]);
               //g[tid*pitch/sizeof(T) + offset + ip] += get_Gradial_d(distance, p[ip*nc + COL_RAD_RS], p[ip*nc + COL_RAD_ETA]);

               //results[threadIdx.x*np + ip] = get_Gradial_d(distance, pShare[ip*nc + COL_RAD_RS], pShare[ip*nc + COL_RAD_ETA]);
               results[threadIdx.x*np + ip] = get_Gradial_d(distance, p[ip*nc + COL_RAD_RS], p[ip*nc + COL_RAD_ETA]);

          }

           __syncthreads();
          for(int ip = 0; ip < np; ip++){
               g[tid*pitch/sizeof(T) + offset + ip] += results[threadIdx.x*np + ip];
          }
          tid += stride;
     }
}

template<typename T>
__global__ void get_Gradial2(T * g0, T * g1, size_t pitch,  T * xyz0, T * xyz1, T * p, size_t np, size_t nc, size_t N, size_t offset){	
	
     __shared__ T xyzShare [TPB*6];
     int tid = threadIdx.x + blockDim.x * blockIdx.x;
     int stride = blockDim.x * gridDim.x;

     while(tid < N){
    
          for(int i = 0; i<3; i++){
               xyzShare[threadIdx.x*6+i*2] = xyz0[N*i+tid];
          }

          for(int j = 0; j<3; j++){
               xyzShare[threadIdx.x*6 + (j*2)+1] = xyz1[N*j+tid];
          }

          __syncthreads();

          T distance = get_dist2<T> (& xyzShare[threadIdx.x*6]);

          for (int ip = 0; ip < np; ip ++){

               //id_cluster = tid
               //offset_by_rel must be passed   
               g0[tid*pitch/sizeof(T) + offset + ip] += get_Gradial_d(distance, p[ip*nc + COL_RAD_RS], p[ip*nc + COL_RAD_ETA]);
               g1[tid*pitch/sizeof(T) + offset + ip] += g0[tid*pitch/sizeof(T) + offset + ip];
          }
          tid += stride;
     }
}

template<typename T>
__global__ void get_Gangular(T * g0, size_t pitch, T * xyz0, T * xyz1, T * xyz2, T * p, size_t np, size_t nc, size_t N, size_t offset){	
     __shared__ T xyzShare [TPB*9];
     int tid = threadIdx.x + blockDim.x * blockIdx.x;
     int stride = blockDim.x * gridDim.x;

     while(tid < N){
    
          for(int i = 0; i<3; i++){
               xyzShare[threadIdx.x*9+i*3] = xyz0[N*i+tid];
          }

          for(int j = 0; j<3; j++){
               xyzShare[threadIdx.x*9 + (j*3)+1] = xyz1[N*j+tid];
          }

          for(int k = 0; k<3; k++){
               xyzShare[threadIdx.x*9 + (k*3)+2] = xyz2[N*k+tid];
          }

          __syncthreads();

          T distance1 = get_dist3<T> (& xyzShare[threadIdx.x*9], 0, 1);
          T distance2 = get_dist3<T> (& xyzShare[threadIdx.x*9], 0, 2);
          T distance3 = get_dist3<T> (& xyzShare[threadIdx.x*9], 1, 2);

          for (int ip = 0; ip < np; ip ++){

               g0[tid*pitch/sizeof(T) + offset + ip] += get_Gangular_d(distance1, distance2, distance3, p[ip*nc + COL_ANG_ETA], 
                                                        p[ip*nc + COL_ANG_ZETA], p[ip*nc + COL_ANG_LAMBD]) ;
          } 

          tid += stride;
     }

}

template <typename T>
class Gfunction_t{
private:
timers_t timers;
timerid_t id;
	 
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

   	matrix_2D_d<T> * xyz_d = new matrix_2D_d<T>(model.NATOM,NCluster*3, xyz);

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

                                   get_Gradial<<< BPG,TPB, np*nc>>> (G_d[atom0_id]->dat,G_d[atom0_id]->pitch, xyz_d->get_elem_ptr(atom0_id,0), 
                                        xyz_d->get_elem_ptr(atom1_id,0), p_d, np, nc, NCluster, offset_by_rel);

						}
                              else if (atom1_id > atom0_id) {

                                   get_Gradial2<<< BPG, TPB >>>(G_d[atom0_id]->dat, G_d[atom1_id]->dat, G_d[atom0_id]->pitch,
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

                                   get_Gangular<<< BPG, TPB >>> (G_d[atom0_id]->dat, G_d[atom0_id]->pitch, xyz_d->get_elem_ptr(atom0_id,0), 
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


     //output comparison

     //get correct output
     T ** correctOutput = nullptr;
     std::string correctFileName = "g_O0.dat";
     read2DArrayfile(correctOutput, G_d[0]->nrow, G_d[0]->ncol, correctFileName.c_str());

     //get g function output from device to host
     memcpy_mtx_d2h(G[0], G_d[0]->dat, G_d[0]->pitch,  G_d[0]->nrow, G_d[0]->ncol);

     for(int i = 0; i< G_d[0]->nrow; i++){
          for(int j = 0; j< G_d[0]->ncol; j++){
               correctOutput[i][j] = abs(correctOutput[i][j] - G[0][i][j]);
          }
     }

std::string filename = "O_out.dat";
G_d[0]->printFile(filename.c_str());
std::cout<<std::endl;

std::ofstream outfile;
std::string filename2 = "diff.dat";
outfile.open(filename2);
for(int j=0;j<G_d[0]->nrow;j++){
     for(int k=0;k<G_d[0]->ncol;k++){
          outfile<<std::setprecision(18)<<std::scientific<<correctOutput[j][k]<<" ";
     }
     outfile<<std::endl;
} 

outfile.close();

std::string filename3 = "O_out_CPU.dat";
outfile.open(filename3);
for(int j=0;j<G_d[0]->nrow;j++){
     for(int k=0;k<G_d[0]->ncol;k++){
          outfile<<std::setprecision(18)<<std::scientific<<G[0][j][k]<<" ";
     }
     outfile<<std::endl;
} 

outfile.close();

delete [] correctOutput;

std::cout<<"End of Gfn"<<std::endl;

}
};
#endif
