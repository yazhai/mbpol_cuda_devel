#ifndef GFUNCTION_H
#define GFUNCTION_H

#include <cstdlib>
#include <vector>
#include <string>
#include <map>
#include <limits>
#include <math.h>
#include <algorithm>

#include "utility.h"
#include "atomTypeID_v2.h"
#include "readGparams_v2.h"

#include "timestamps.h"

// Define the cblas library 
#ifdef _USE_GSL
#include <gsl/gsl_cblas.h>
#elif _USE_MKL
#include <mkl_cblas.h>
#else 
//#include <gsl/gsl_cblas.h>
#endif


#ifdef _OPENMP
#include <omp.h>
#endif 


const int COL_RAD_RS = 3;
const int COL_RAD_ETA = 2; 
const int COL_RAD_TOTAL = 5;

const int COL_ANG_ETA = 2;
const int COL_ANG_ZETA = 4;
const int COL_ANG_LAMBD=3;
const int COL_ANG_TOTAL=6;



template <typename T>
class Gfunction_t{
private:

//===========================================================================================
//
// Matrix Elementary functions
//T cutoff(T R, T R_cut = 10);
//T get_cos(T Rij, T Rik, T Rjk);
//T get_Gradial(T Rs, T eta, T  Rij);
//T get_Gangular(T Rij, T Rik, T Rjk, T eta, T zeta, T lambd);

T cutoff(T R, T R_cut=10.0) {
    T f=0.0;
    if (R < R_cut) {    
        //T t =  tanh(1.0 - R/R_cut) ;   // avoid using `tanh`, which costs more than `exp` 
        T t =  1.0 - R/R_cut;        
        t = exp(2*t);
        t = (t-1) / (t+1);                
        f = t * t * t ;        
    }
    return f ;
}


T get_cos(T Rij, T Rik, T Rjk) {
    //cosine of the angle between two vectors ij and ik    
    T Rijxik = Rij*Rik ;    
    if ( Rijxik != 0 ) {
          return ( ( Rij*Rij + Rik*Rik - Rjk*Rjk )/ (2.0 * Rijxik) );
    } else {
          return  std::numeric_limits<T>::infinity();
    }
}


T get_Gradial(T  Rij, T Rs, T eta){
     // T G_rad = cutoff(Rij);    
     T G_rad = 1.0 ; // cutoff fucntion is switched off according to the update of physical model
     if ( G_rad > 0 ) {
          G_rad *= exp( -eta * ( (Rij-Rs)*(Rij-Rs) )  )  ;
     }
     return G_rad;
}


T get_Gangular(T Rij, T Rik, T Rjk, T eta, T zeta, T lambd){    
    // T G_ang = cutoff(Rij)*cutoff(Rik)*cutoff(Rjk);   
     T G_ang = 1.0;
    if ( G_ang > 0) {    
          G_ang *=   2 * pow( (1.0 + lambd* get_cos(Rij, Rik, Rjk))/2.0, zeta) 
                     * exp(-eta*  ( (Rij+Rik+Rjk)*(Rij+Rik+Rjk) ) );    
    } 
    return G_ang ;    
}


T get_dist(size_t atom1, size_t atom2, size_t dimer_idx){
     T a = xyz[dimer_idx][3*atom1] - xyz[dimer_idx][3*atom2];
     T b = xyz[dimer_idx][3*atom1+1] - xyz[dimer_idx][3*atom2+1];
     T c = xyz[dimer_idx][3*atom1+2] - xyz[dimer_idx][3*atom2+2];
     return sqrt(a*a + b*b + c*c);
}


//===========================================================================================
// Vectorized functions
//
// These functions are not vectorized at the moment, 
// but API are left as vectorized form for consecutive memory utilization and 
// future compatible possibility with other linear algebra libraries.
// 
// Vectorized functions list:
//void cutoff(T* & Rdst, T* & Rrsc, size_t n, T R_cut=10);
//void get_cos(T * & Rdst, T * & Rij, T * & Rik, T * & Rjk, size_t n);
//void get_Gradial(T* & Rdst, T* & Rij, size_t n, T Rs, T eta);
//void get_Gangular(T* & Rdst, T* & Rij, T* & Rik, T* & Rjk, size_t n, T eta,T zeta, T lambd );
//void get_Gradial_add(T* & Rdst, T* & tmp, T* & Rij, size_t n, T Rs, T eta);
//void get_Gangular_add(T* & Rdst, T*& tmp, T* & Rij, T* & Rik, T* & Rjk, size_t n, T eta,T zeta, T lambd );


void cutoff(T* rst, T* Rij, size_t n, T R_cut=10) {    
#ifdef _OPENMP
#pragma omp parallel for simd shared(rst, Rij, R_cut, n)
#endif    
    for (int i=0; i<n; i++){
          rst[i] = cutoff(Rij[i], R_cut);
    }             
};



void get_cos(T * rst, T * Rij, T * Rik, T * Rjk, size_t n) {
#ifdef _OPENMP
#pragma omp parallel for simd shared(rst, Rij, Rik, Rjk)
#endif
  for (int i=0; i<n; i++){     
     rst[i] = get_cos(Rij[i], Rik[i], Rjk[i]);  
  }
};



void get_Gradial(T* rst, T* Rij, size_t n, T Rs, T eta, T R_cut=10 ){ 
  // cutoff(rst, Rij, n, R_cut);
#ifdef _OPENMP
#pragma omp parallel for simd shared(rst, Rij, Rs, eta)
#endif  
  for (int i=0; i<n ; i++) {
     //rst[i] = cutoff(Rij[i]);  // Use vectorized cutoff function instead
     //if (rst[i] >0){    
          rst[i] = exp( -eta * ( (Rij[i]-Rs)*(Rij[i]-Rs) )  )  ;
     //}
  } 
};



void get_Gradial_add(T* rst, T* Rij, size_t n, T Rs, T eta , T* tmp = nullptr ){
     bool iftmp = false;
     if (tmp == nullptr){
          tmp = new T[n]();
          iftmp = true;             
     }          
     get_Gradial(tmp, Rij, n, Rs, eta);
     #ifdef _OPENMP
     #pragma omp parallel for simd shared(rst, tmp, n)
     #endif            
     for (int ii=0; ii<n; ii++){
          rst[ii] += tmp[ii] ;
     }  
     if (iftmp) delete[] tmp;          
};
 



void get_Gangular(T* rst, T* Rij, T* Rik, T* Rjk, size_t n, T eta, T zeta, T lambd ){
#ifdef _OPENMP
#pragma omp parallel for simd shared(rst, Rij, Rik, Rjk, eta, zeta, lambd)
#endif
  for (int i=0; i<n; i++){
    rst[i]=get_Gangular(Rij[i], Rik[i], Rjk[i], eta, zeta, lambd);
  }
};


void get_Gangular_add(T* rst, T* Rij, T* Rik, T* Rjk, size_t n, T eta, T zeta, T lambd, T* tmp = nullptr ){
     bool iftmp = false;
     if (tmp == nullptr){
          tmp = new T[n]();
          iftmp = true;             
     }     
     get_Gangular(tmp, Rij, Rik, Rjk, n, eta, zeta, lambd);
     #ifdef _OPENMP
     #pragma omp parallel for simd shared(rst, tmp, n)
     #endif            
     for (int ii=0; ii<n; ii++){
          rst[ii] += tmp[ii] ;
     }  
     if (iftmp) delete[] tmp;
};


//New Cutoff Functions

//SIMPLE IMPLEMENTATION -- Can improve Later
T base_fswitch(T ri,T rf,T r){
     T value;
     T coef = M_PI/(rf-ri);
     T temp = (1.0 + cos(coef*(r-ri)))/2.0;
	if(r<rf)
		value = (r>=ri)? temp : 1;
	else	
		value = 0;

     return value;
}

void fswitch_2b(T ri, T rf) {
    T dist;
    for(int i=0;i<NCluster;i++){
        dist = get_dist(0,1,i);
        cutoffs[i] = base_fswitch(ri,rf,dist);
    }

}

void fswitch_3b(T ri, T rf) {
    T dist1,dist2,dist3;
    for(int i=0;i<NCluster;i++){
        dist1 = get_dist(0,1,i);
        dist2 = get_dist(0,2,i);
        dist3 = get_dist(1,2,i);
        cutoffs[i] = base_fswitch(ri,rf,dist1)*base_fswitch(ri,rf,dist2)*base_fswitch(ri,rf,dist3);
    }
}




// Timers for benchmarking
timers_t timers;
timerid_t id, id1, id2 , id3;

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
std::vector<T> G_param_max_size;        //max size of parameters for each atom type



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
          }
          G_param_max_size.push_back(s);
     };
};

void load_cutoffs(){
    T ri = 4.5, rf = 6.5;
    if(cutoffs != nullptr)  delete [] cutoffs;
    cutoffs = new T[NCluster];

    if(TypeNAtom[0] == 2){      //if 2 body
        fswitch_2b(4.5,6.5);
    }
    else{
        fswitch_3b(0,4.5);      //else 3 body
    }

    return;
}



//=================================================================================
// G-function Construction
//void make_G();
//void make_G(const char* _distfile, int _titleline, const char* _colidxfile, const char* _paramfile, const char* _ordfile);

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

     // init atom count and index in each type
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

for(int loop = 0; loop < 1; loop++){

     timers.insert_random_timer(id3, 1 , "Gf_run_all");
     timers.timer_start(id3);     
       
     // if(!model.XYZ_TRANS) model.transpose_xyz();
     xyz = model.XYZ ; // Give reference to xyz
     
      
     size_t Cluster_Batch = NCluster ;  // 
     // size_t id_cluster = 0 ;

// The most outer iteration is looping for every Cluster_Batch dimers/trimers
// while the most inner iteration is looping for every cluster within the Cluster_Batch
for (size_t batchstart = 0; batchstart < NCluster; batchstart += Cluster_Batch){
      
      size_t batchlimit = std::min(batchstart+Cluster_Batch, NCluster);

          // for each type (first type in seq)
          for(idx_t type_id = 0; type_id < NType; type_id ++ ){

               // iterate through all paramter seq: for "H", iterate in sequence such as "HH"-"HO"-"HHO"-"HHH" depending on the seq defined
               idx_t relation_idx = 0;    // current relation writing to this index in G
               size_t offset_by_rel = 0;  // Index already finished

               for (auto rel = model.seq[type_id].begin(); rel != model.seq[type_id].end(); rel++ ){

                    size_t np = gparams.PARAMS[*rel].nparam;  // number of parameter pairs
                    size_t nc = gparams.PARAMS[*rel].ncol;  // number of param in one line
                    T * p = (&gparams.PARAMS[*rel].dat_h[0]); // params

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
}    // end of  Cluster_Batch 

     timers.timer_end(id3);

}
     timers.get_all_timers_info();
     timers.get_time_collections();



};


void scaleG(const char * scaleFolder){
    size_t i = 0, dim;
    T** scaleMatrix = nullptr;
    std::string atomType, atomNum, fileName;
    size_t single = 1;
    for (auto it = G.begin(); it != G.end(); it++){
        dim = G_param_max_size[model.TYPE_EACHATOM[i]];
        init_mtx_in_mem<T>(scaleMatrix,dim,single);
        atomType = std::to_string(model.TYPE_EACHATOM[i]);
         atomNum = std::to_string(i);
        fileName = scaleFolder + atomType + atomNum + "_max";
        std::cout<<fileName<<std::endl;
        read2DArrayfile<T>(scaleMatrix, dim, single, fileName.c_str());
        for(int ii=0; ii<NCluster;ii++){
               for(int jj=0;jj<dim;jj++){
                    (*it)[ii][jj] /= scaleMatrix[jj][0];
            }
        }
        i++;
    }

}

};

#endif
