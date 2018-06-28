

#include <limits>
#include <cstdlib>
#include <iomanip>
#include <utility>
#include <cstddef>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <cstdint>
#include <limits>
#include <cstring>
#include <algorithm>
#include <iomanip>

#include <iostream>
#include <sstream>
#include <fstream>

#include <vector>
#include <map>
#include <string>

#include "Gfunction_v2.h"
#include "H_2H2O_max"
#include "O_2H2O_max"

#include "H_3H2O_max"
#include "O_3H2O_max"


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

using namespace MBbpnnPlugin;

// ====================================================================
// Gfunction_t private methods
template <typename T>
T Gfunction_t<T>::cutoff(T R, T R_cut) {
    T f=0.0;
    if (R < R_cut) {    
        //T t =  tanh(1.0 - R/R_cut) ;   // avoid using `tanh`, which costs more than `exp` 
        T t =  1.0 - R/R_cut;        
        t = exp(2*t);
        t = (t-1) / (t+1);                
        f = t * t * t ;        
    }
    return f ;
};


template <typename T>
T Gfunction_t<T>::get_cos(T Rij, T Rik, T Rjk) {
    //cosine of the angle between two vectors ij and ik    
    T Rijxik = Rij*Rik ;    
    if ( Rijxik != 0 ) {
          return ( ( Rij*Rij + Rik*Rik - Rjk*Rjk )/ (2.0 * Rijxik) );
    } else {
          return  std::numeric_limits<T>::infinity();
    }
}


template <typename T>
T Gfunction_t<T>::get_Gradial(T  Rij, T Rs, T eta){
     // T G_rad = cutoff(Rij);    
     T G_rad = 1.0 ; // cutoff fucntion is switched off according to the update of physical model
     if ( G_rad > 0 ) {
          G_rad *= exp( -eta * ( (Rij-Rs)*(Rij-Rs) )  )  ;
     }
     return G_rad;
}

template <typename T>
T Gfunction_t<T>::get_Gangular(T Rij, T Rik, T Rjk, T eta, T zeta, T lambd){    
    // T G_ang = cutoff(Rij)*cutoff(Rik)*cutoff(Rjk);   
     T G_ang = 1.0;
    if ( G_ang > 0) {    
          G_ang *=   2 * pow( (1.0 + lambd* get_cos(Rij, Rik, Rjk)) * 0.5, zeta) 
                     * exp(-eta*  ( (Rij+Rik+Rjk)*(Rij+Rik+Rjk) ) );    
    } 
    return G_ang ;     
}

template <typename T>
T Gfunction_t<T>::get_dist(size_t atom1, size_t atom2, size_t dimer_idx){
     T a, b, c;
     a = xyz[3*atom1  ][dimer_idx] - xyz[3*atom2  ][dimer_idx];
     b = xyz[3*atom1+1][dimer_idx] - xyz[3*atom2+1][dimer_idx];
     c = xyz[3*atom1+2][dimer_idx] - xyz[3*atom2+2][dimer_idx];          
     return sqrt(a*a + b*b + c*c);
}


template <typename T>
void Gfunction_t<T>::cutoff(T* rst, T* Rij, size_t n, T R_cut) {    
#ifdef _OPENMP
#pragma omp parallel for simd shared(rst, Rij, R_cut, n)
#endif    
    for (size_t i=0; i<n; i++){
          rst[i] = cutoff(Rij[i], R_cut);
    }             
};


template <typename T>
void Gfunction_t<T>::get_cos(T * rst, T * Rij, T * Rik, T * Rjk, size_t n) {
#ifdef _OPENMP
#pragma omp parallel for simd shared(rst, Rij, Rik, Rjk)
#endif
  for (size_t i=0; i<n; i++){     
     rst[i] = get_cos(Rij[i], Rik[i], Rjk[i]);  
  }
};



template <typename T>
void Gfunction_t<T>::get_Gradial(T* rst, T* Rij, size_t n, T Rs, T eta, T R_cut){ 
  // cutoff(rst, Rij, n, R_cut);
#ifdef _OPENMP
#pragma omp parallel for simd shared(rst, Rij, Rs, eta)
#endif  
  for (size_t i=0; i<n ; i++) {
     //rst[i] = cutoff(Rij[i]);  // Use vectorized cutoff function instead
     //if (rst[i] >0){    
          rst[i] = exp( -eta * ( (Rij[i]-Rs)*(Rij[i]-Rs) )  )  ;
     //}
  } 
};


template <typename T>
void Gfunction_t<T>::get_Gradial_add(T* rst, T* Rij, size_t n, T Rs, T eta , T* tmp){
     bool iftmp = false;
     if (tmp == nullptr){
          tmp = new T[n]();
          iftmp = true;             
     }          
     get_Gradial(tmp, Rij, n, Rs, eta);
     #ifdef _OPENMP
     #pragma omp parallel for simd shared(rst, tmp, n)
     #endif            
     for (size_t ii=0; ii<n; ii++){
          rst[ii] += tmp[ii] ;
     }  
     if (iftmp) delete[] tmp;          
};
 



template <typename T>
void Gfunction_t<T>::get_Gangular(T* rst, T* Rij, T* Rik, T* Rjk, size_t n, T eta, T zeta, T lambd ){
#ifdef _OPENMP
#pragma omp parallel for simd shared(rst, Rij, Rik, Rjk, eta, zeta, lambd)

#endif
  for (size_t i=0; i<n; i++){
    rst[i]=get_Gangular(Rij[i], Rik[i], Rjk[i], eta, zeta, lambd);
  }
};

template <typename T>
void Gfunction_t<T>::get_Gangular_add(T* rst, T* Rij, T* Rik, T* Rjk, size_t n, T eta, T zeta, T lambd, T* tmp){
     bool iftmp = false;
     if (tmp == nullptr){
          tmp = new T[n]();
          iftmp = true;             
     }     
     get_Gangular(tmp, Rij, Rik, Rjk, n, eta, zeta, lambd);
     #ifdef _OPENMP
     #pragma omp parallel for simd shared(rst, tmp, n)
     #endif            
     for (size_t ii=0; ii<n; ii++){
          rst[ii] += tmp[ii] ;
     }  
     if (iftmp) delete[] tmp;
};


// Helper functions for gradient: 
//
// e.g. get_dGdD = local gradient dG/dD
//
template <typename T>
T Gfunction_t<T>::get_dGdR_rad(T Rij, T Rref, T eta){
     // for gradient in radial
     T d = Rij - Rref;
     return exp( - eta * d * d) * ( -2 * eta * d) ;
}


template <typename T>
T Gfunction_t<T>::get_dGdR_ang_IJ(T Rij, T Rik, T Rjk, T eta, T zeta, T lambd){
     T Rij2 = Rij*Rij;
     T Rik2 = Rik*Rik;
     T Rjk2 = Rjk*Rjk;

     T ACOS = (Rij2 + Rik2 - Rjk2) / (2 * Rij * Rik);  

     T dACOSdRij = 1 - ( Rij2 + Rik2  - Rjk2) / (2*Rij2);
     dACOSdRij /= Rik;

     T t1 = 1+ lambd * ACOS ;
     T t2 = pow( t1 * 0.5, zeta) ;

     // Here using exp(-eta*(Rij2 + Rik2 + Rjk2))
     // T t3 = exp(-eta * (Rij2 + Rik2 + Rjk2) );
     // return (zeta * lambd * dACOSdRij / t1 - 2 * eta * Rij)  * 2 * t2 * t3 ;

     // Here using exp(-eta*(Rij + Rik + Rjk)**2 )
     T t3 = exp(-eta* (Rij+Rik+Rjk) * (Rij+Rik+Rjk) ) ;
     return (zeta * lambd * dACOSdRij / t1 - 2 * eta * (Rij+Rik+Rjk) )  * 2 * t2 * t3 ;
}


template <typename T>
T Gfunction_t<T>::get_dGdR_ang_JK(T Rij, T Rik, T Rjk, T eta, T zeta, T lambd){
     T Rij2 = Rij*Rij;
     T Rik2 = Rik*Rik;
     T Rjk2 = Rjk*Rjk;

     T ACOS = (Rij2 + Rik2 - Rjk2) / (2 * Rij * Rik);  

     T dACOSdRjk = - ( Rjk ) / (Rij * Rik);

     T t1 = 1+ lambd * ACOS ;
     T t2 = pow( t1 * 0.5, zeta) ;

     // Here using exp(-eta*(Rij2 + Rik2 + Rjk2))
     // T t3 = exp(-eta * (Rij2 + Rik2 + Rjk2) );
     // return (zeta * lambd * dACOSdRij / t1 - 2 * eta * Rij)  * 2 * t2 * t3 ;

     // Here using exp(-eta*(Rij + Rik + Rjk)**2 )
     T t3 = exp(-eta* (Rij+Rik+Rjk) * (Rij+Rik+Rjk) ) ;
     return (zeta * lambd * dACOSdRjk / t1 - 2 * eta * (Rij+Rik+Rjk) )  * 2 * t2 * t3 ;     
}


template <typename T>
T Gfunction_t<T>::get_dRdX(T Rij, T Xi, T Xj){
     return (Xi - Xj) / Rij ;
}


template <typename T>
void Gfunction_t<T>::get_dRdX_byID(T *  dRdX ,  idx_t atom1, idx_t atom2, size_t dimer_idx){
     T R = get_dist(atom1, atom2, dimer_idx);

     dRdX[0] = get_dRdX(R, xyz[3*atom1  ][dimer_idx]  , xyz[3*atom2  ][dimer_idx]  ) ;
     dRdX[1] = get_dRdX(R, xyz[3*atom1+1][dimer_idx]  , xyz[3*atom2+1][dimer_idx]  ) ;
     dRdX[2] = get_dRdX(R, xyz[3*atom1+2][dimer_idx]  , xyz[3*atom2+2][dimer_idx]  ) ;          
}


template <typename T>
void Gfunction_t<T>::load_resv_scales(std::string tag){
     // this function loads the default scaling factor data into the reserved mappings.
     size_t N ;
     T* PT = nullptr;

     if (tag == "2h2o"){
          // the following code saves the default "H" scaler
          N = (size_t) ( sizeof(H_2H2O_MAX) / sizeof(double) );
          PT = new T[N];
          if ( G_SCALE_CONST.find("H_2H2O") == G_SCALE_CONST.end() ){
               for(size_t i = 0; i< N; i++){
                    PT[i] = (T) (1 / H_2H2O_MAX[i]) ;
               };          
               G_SCALE_CONST["H_2H2O"] = PT;
          }

          // the following code saves the default "O" scaler
          N = (size_t) ( sizeof(O_2H2O_MAX) / sizeof(double) );
          PT = new T[N];
          if ( G_SCALE_CONST.find("O_2H2O") == G_SCALE_CONST.end() ){
               for(size_t i = 0; i< N; i++){
                    PT[i] = (T) (1 / O_2H2O_MAX[i]) ;
               };          
               G_SCALE_CONST["O_2H2O"] = PT;
          }
     } else if (tag =="3h2o") {
          // the following code saves the default "H" scaler
          N = (size_t) ( sizeof(H_3H2O_MAX) / sizeof(double) );
          PT = new T[N];
          if ( G_SCALE_CONST.find("H_3H2O") == G_SCALE_CONST.end() ){
               for(size_t i = 0; i< N; i++){
                    PT[i] = (T) (1 / H_3H2O_MAX[i]) ;
               };          
               G_SCALE_CONST["H_3H2O"] = PT;
          }

          // the following code saves the default "O" scaler
          N = (size_t) ( sizeof(O_3H2O_MAX) / sizeof(double) );
          PT = new T[N];
          if ( G_SCALE_CONST.find("O_3H2O") == G_SCALE_CONST.end() ){
               for(size_t i = 0; i< N; i++){
                    PT[i] = (T) (1 / O_3H2O_MAX[i]) ;
               };          
               G_SCALE_CONST["O_3H2O"] = PT;
          }
     }
}



template <typename T>
void Gfunction_t<T>::init_G(){
     // some preparation
     for(auto it = G.begin(); it!= G.end(); it++){
          clearMemo(*it);
     }
     G.clear();
     G_param_max_size.clear();
     for(idx_t i = 0; i< this->NATOM; i++){
          T** tmp = nullptr;
          G.push_back(tmp);
     }
     
     for(idx_t type_id = 0; type_id < this->NTYPE; type_id ++ ){
          // get how many params for one type
          size_t s = 0;
          for(auto it =this->seq[type_id].begin(); it!=this->seq[type_id].end() ; it++){
               s += gparams.PARAMS[*it].nparam ;
          };
          for(auto it = this->ATOMS[type_id].begin(); it!= this->ATOMS[type_id].end(); it++){
               init_mtx_in_mem(G[*it], s, this->NCLUSTER);
          };

          G_param_max_size.push_back(s);

     };

     if( xyz != nullptr) {
          clearMemo(xyz);
          xyz = nullptr;
     }

     
     if( this->XYZ_ONE_ATOM_PER_COL ){
          init_mtx_in_mem(xyz, this->NATOM*3, this->NCLUSTER);
          std::copy(this->XYZ[0], this->XYZ[0]+ this->NATOM*this->NCLUSTER*3, xyz[0]);
     } else{
           this->transpose_xyz(xyz);
     }

};


template <typename T>
void Gfunction_t<T>::scale_G(std::string tag){
     if(  ( G_scale.empty() ) && (  this->get_type_idx("O", false) != DEFAULT_ID ) && ( this->get_type_idx("H", false) != DEFAULT_ID ) ){

          if (tag == "2h2o"){
               if (this->TYPE_INDEX[0] == "O" ) {
                    G_scale.push_back(G_SCALE_CONST["O_2H2O"] );
                    G_scale.push_back(G_SCALE_CONST["H_2H2O"] );
               } else {
                    G_scale.push_back(G_SCALE_CONST["H_2H2O"] );
                    G_scale.push_back(G_SCALE_CONST["O_2H2O"] );          
               }
          } else if (tag == "3h2o"){
               if (this->TYPE_INDEX[0] == "O" ) {
                    G_scale.push_back(G_SCALE_CONST["O_3H2O"] );
                    G_scale.push_back(G_SCALE_CONST["H_3H2O"] );
               } else {
                    G_scale.push_back(G_SCALE_CONST["H_3H2O"] );
                    G_scale.push_back(G_SCALE_CONST["O_3H2O"] );          
               }               
          }
     }

     idx_t atom_type = 0;
     for(auto atoms_it = this->ATOMS.begin(); atoms_it != this->ATOMS.end(); atoms_it++){
          // for a specific atom type, find all the this->ATOMS
          // scale these G results with G_scale
          for (auto it = (*atoms_it).begin(); it!= (*atoms_it).end(); it++){
               T** g = G[*it] ;
               size_t ncluster = this->NCLUSTER;
               size_t pmax = G_param_max_size[atom_type];
               T* gsc = G_scale[atom_type];
#ifdef _OPENMP
#pragma omp parallel for simd shared(g, ncluster,  gsc, pmax)
#endif 
for(size_t clusterid=0; clusterid < ncluster; clusterid++){
               for (size_t p = 0; p < pmax ; p++){    
                    g[p][clusterid] = g[p][clusterid] * gsc[p];
               }
}
          }
          atom_type ++ ;
     }
};



template <typename T>
void Gfunction_t<T>::scale_G_grd(std::vector<T**> dfdG){
     idx_t atom_type = 0;
     for(auto atoms_it = this->ATOMS.begin(); atoms_it != this->ATOMS.end(); atoms_it++){
          // for a specific atom type, find all the this->ATOMS
          // scale these G results with G_scale
          for (auto it = (*atoms_it).begin(); it!= (*atoms_it).end(); it++){
               T** dg = dfdG[*it] ;
               size_t ncluster = this->NCLUSTER;
               size_t pmax = G_param_max_size[atom_type];
               T* gsc =G_scale[atom_type];
#ifdef _OPENMP
#pragma omp parallel for simd shared(dg, ncluster,  gsc, pmax)
#endif 
for(size_t clusterid=0; clusterid < ncluster ; clusterid++){
               for (size_t p = 0; p < pmax; p++){
                    dg[p][clusterid] = dg[p][clusterid] * gsc[p] ;
               };
};
          };
          atom_type ++ ;
     };
};



template <typename T>
T Gfunction_t<T>::base_fswitch(T ri,T rf,T r){
     T value = 0;
	if(r<rf) {
          T coef = M_PI/(rf-ri); 
          T temp = (1.0 + cos(coef*(r-ri)))/2.0;     
		value = (r>=ri)? temp : 1; 
     }
     return value;
}


template <typename T>
T Gfunction_t<T>::base_dswitch(T ri,T rf,T r){
     T value = 0;
	if ( (r<rf) && ( r> ri ) ) {
          T coef = M_PI/(rf-ri); 
          value = ( - sin(coef*(r-ri))* coef )/2.0;  
     }
     return value;
}



template <typename T>
void Gfunction_t<T>::fswitch_2b(T ri, T rf, idx_t at0, idx_t at1) {
     size_t ncluster = this->NCLUSTER;
     T* sw = switch_factor;
#ifdef _OPENMP
#pragma omp parallel for simd shared(ncluster, sw, ri, rf)
#endif  
     for(size_t i=0;i<ncluster;i++){
          T dist = get_dist(at0,at1,i);
          sw[i] = base_fswitch(ri,rf,dist);
     }
}

template <typename T>
void Gfunction_t<T>::fswitch_3b(T ri, T rf, idx_t at0, idx_t at1, idx_t at2) {
     size_t ncluster = this->NCLUSTER;
     T* sw = switch_factor;
#ifdef _OPENMP
#pragma omp parallel for simd shared(ncluster, sw, ri, rf)
#endif      
     for(size_t i=0;i<ncluster;i++){
          T s01 = base_fswitch(ri,rf,get_dist(at0,at1,i));
          T s02 = base_fswitch(ri,rf,get_dist(at0,at2,i));
          T s12 = base_fswitch(ri,rf,get_dist(at1,at2,i));
          sw[i] = s01*s02 + s01*s12 + s02*s12;
     }
}



template <typename T>
void Gfunction_t<T>::fswitch_with_grad_2b(T ri, T rf, idx_t at0, idx_t at1) {
     size_t ncluster = this->NCLUSTER;
     T* sw = switch_factor;
     T** dsw = dswitchdx;
#ifdef _OPENMP
#pragma omp parallel for simd shared(ncluster, sw, dsw, ri, rf, at0, at1)
#endif  
     for(size_t i=0;i<ncluster;i++){
          T dist = get_dist(at0,at1,i);
          T dRdx[3];
          get_dRdX_byID(&(dRdx[0]), at0, at1, i);
          sw[i] = base_fswitch(ri,rf,dist);
          T dswitch = base_dswitch(ri,rf,dist);
          
          dsw[0][i] = dswitch * dRdx[0] ;
          dsw[1][i] = dswitch * dRdx[1] ; 
          dsw[2][i] = dswitch * dRdx[2] ;
          dsw[3][i] =-dswitch * dRdx[0] ;
          dsw[4][i] =-dswitch * dRdx[1] ; 
          dsw[5][i] =-dswitch * dRdx[2] ;
     }
}

template <typename T>
void Gfunction_t<T>::fswitch_with_grad_3b(T ri, T rf, idx_t at0, idx_t at1, idx_t at2) {
     size_t ncluster = this->NCLUSTER;
     T* sw = switch_factor;
     T** dsw = dswitchdx ;
#ifdef _OPENMP
#pragma omp parallel for simd shared(ncluster, sw, dsw, ri, rf, at0, at1, at2)
#endif      
     for(size_t i=0;i<ncluster;i++){
          T r01 = get_dist(at0,at1,i);
          T r02 = get_dist(at0,at2,i);
          T r12 = get_dist(at1,at2,i);

          T s01 = base_fswitch(ri,rf,r01);
          T s02 = base_fswitch(ri,rf,r02);
          T s12 = base_fswitch(ri,rf,r12);

          T ds01 = base_dswitch(ri, rf, r01);
          T ds02 = base_dswitch(ri, rf, r02);
          T ds12 = base_dswitch(ri, rf, r12);

          sw[i] = s01*s02 + s01*s12 + s02*s12;

          T dRdx[3][3] ;
          get_dRdX_byID(&(dRdx[0][0]), at0, at1, i);
          get_dRdX_byID(&(dRdx[1][0]), at0, at2, i);   
          get_dRdX_byID(&(dRdx[2][0]), at1, at2, i);

          T dsdr01 = (s02 + s12) * ds01;
          T dsdr02 = (s01 + s12) * ds02;
          T dsdr12 = (s01 + s02) * ds12;

          // dsw at0
          dsw[0][i] = dsdr01 * dRdx[0][0] + dsdr02 * dRdx[1][0];
          dsw[1][i] = dsdr01 * dRdx[0][1] + dsdr02 * dRdx[1][1];
          dsw[2][i] = dsdr01 * dRdx[0][2] + dsdr02 * dRdx[1][2];                    
          // dsw at1
          dsw[3][i] =-dsdr01 * dRdx[0][0] + dsdr12 * dRdx[2][0];
          dsw[4][i] =-dsdr01 * dRdx[0][1] + dsdr12 * dRdx[2][1];
          dsw[5][i] =-dsdr01 * dRdx[0][2] + dsdr12 * dRdx[2][2];          

          // dsw at2
          dsw[6][i] =-dsdr02 * dRdx[1][0] - dsdr12 * dRdx[2][0];
          dsw[7][i] =-dsdr02 * dRdx[1][1] - dsdr12 * dRdx[2][1];
          dsw[8][i] =-dsdr02 * dRdx[1][2] - dsdr12 * dRdx[2][2];             
     }
}

template <typename T>
void Gfunction_t<T>::cal_switch(int flag, idx_t at0, idx_t at1, idx_t at2){
    const T r1 = 4.5, r2 = 6.5, r3 = 0;
    if(switch_factor != nullptr)  delete [] switch_factor;
    switch_factor = new T[this->NCLUSTER];
    
    // hard coded for 2B/3B H2O
    if(flag == 2){      //if 2 boby 
          fswitch_2b(r1, r2, at0, at1);
    }
    else{ //else 3 body
          fswitch_3b(r3, r1, at0, at1, at2);
    }
    return;
}


template <typename T>
void Gfunction_t<T>::cal_switch_with_grad(int flag, idx_t at0, idx_t at1, idx_t at2){
    const T r1 = 4.5, r2 = 6.5, r3 = 0;
    if(switch_factor != nullptr)  delete [] switch_factor;
    switch_factor = new T[this->NCLUSTER];
    
    // hard coded for 2B/3B H2O
    if(flag == 2){      //if 2 boby 
          init_mtx_in_mem(dswitchdx, 6, this->NCLUSTER);
          fswitch_with_grad_2b(r1, r2, at0, at1);
    }
    else{ //else 3 body
          init_mtx_in_mem(dswitchdx, 9, this->NCLUSTER);
          fswitch_with_grad_3b(r3, r1, at0, at1, at2);
    }
    return;
}



template <typename T>
void Gfunction_t<T>::scale_dfdx_with_switch(int flag, T* e, idx_t at0, idx_t at1, idx_t at2){

     T** dfdx = dfdxyz ;
     T** dsw  = dswitchdx ;
     T*  s = switch_factor ;
     size_t N = this->NCLUSTER;    

    if(flag == 2){      //if 2 body   
          for(idx_t at = 0; at < this->NATOM; at++){
               if ( at == at0 ) {  // if the atom is involved with switching factor calculation, scale the dfdx and then add the gradient from   
#ifdef _OPENMP
#pragma omp parallel for simd shared(e, at0, at1, at2, dfdx, dsw, s)
#endif          
                    for(size_t i=0; i < N; i++){
                         dfdx[3*at    ][i] *= s[i]; 
                         dfdx[3*at + 1][i] *= s[i]; 
                         dfdx[3*at + 2][i] *= s[i]; 
                         dfdx[3*at    ][i] += e[i] * dsw[0][i]; 
                         dfdx[3*at + 1][i] += e[i] * dsw[1][i]; 
                         dfdx[3*at + 2][i] += e[i] * dsw[2][i];
                    }
               } else if( at == at1 ) { 
#ifdef _OPENMP
#pragma omp parallel for simd shared(e, at0, at1, at2, dfdx, dsw, s)
#endif          
                    for(size_t i=0; i < N; i++){
                         dfdx[3*at    ][i] *= s[i]; 
                         dfdx[3*at + 1][i] *= s[i]; 
                         dfdx[3*at + 2][i] *= s[i]; 
                         dfdx[3*at    ][i] += e[i] * dsw[3][i]; 
                         dfdx[3*at + 1][i] += e[i] * dsw[4][i]; 
                         dfdx[3*at + 2][i] += e[i] * dsw[5][i];
                    }
               } else { // if the atom is not involved with switching factor calculation, scale the dfdx with switching factor
#ifdef _OPENMP
#pragma omp parallel for simd shared(e, at0, at1, at2, dfdx, dsw, s)
#endif          
                    for(size_t i=0; i < N; i++){
                         dfdx[3*at    ][i] *= s[i]; 
                         dfdx[3*at + 1][i] *= s[i]; 
                         dfdx[3*at + 2][i] *= s[i];             
                    }
               }
          }
    }
    else{
          for(idx_t at = 0; at < this->NATOM; at++){
               if ( at == at0 ) {  // if the atom is involved with switching factor calculation, scale the dfdx and then add the gradient from   
#ifdef _OPENMP
#pragma omp parallel for simd shared(e, at0, at1, at2, dfdx, dsw, s)
#endif          
                    for(size_t i=0; i < N; i++){
                         dfdx[3*at    ][i] *= s[i]; 
                         dfdx[3*at + 1][i] *= s[i]; 
                         dfdx[3*at + 2][i] *= s[i]; 
                         dfdx[3*at    ][i] += e[i] * dsw[0][i]; 
                         dfdx[3*at + 1][i] += e[i] * dsw[1][i]; 
                         dfdx[3*at + 2][i] += e[i] * dsw[2][i];
                    }
               } else if( at == at1 ) { 
#ifdef _OPENMP
#pragma omp parallel for simd shared(e, at0, at1, at2, dfdx, dsw, s)
#endif          
                    for(size_t i=0; i < N; i++){
                         dfdx[3*at    ][i] *= s[i]; 
                         dfdx[3*at + 1][i] *= s[i]; 
                         dfdx[3*at + 2][i] *= s[i]; 
                         dfdx[3*at    ][i] += e[i] * dsw[3][i]; 
                         dfdx[3*at + 1][i] += e[i] * dsw[4][i]; 
                         dfdx[3*at + 2][i] += e[i] * dsw[5][i];
                    }
               } else if( at == at2 ) { 
#ifdef _OPENMP
#pragma omp parallel for simd shared(e, at0, at1, at2, dfdx, dsw, s)
#endif          
                    for(size_t i=0; i < N; i++){
                         dfdx[3*at    ][i] *= s[i]; 
                         dfdx[3*at + 1][i] *= s[i]; 
                         dfdx[3*at + 2][i] *= s[i]; 
                         dfdx[3*at    ][i] += e[i] * dsw[6][i]; 
                         dfdx[3*at + 1][i] += e[i] * dsw[7][i]; 
                         dfdx[3*at + 2][i] += e[i] * dsw[8][i];
                    }                    
               } else { // if the atom is not involved with switching factor calculation, scale the dfdx with switching factor
#ifdef _OPENMP
#pragma omp parallel for simd shared(e, at0, at1, at2, dfdx, dsw, s)
#endif          
                    for(size_t i=0; i < N; i++){
                         dfdx[3*at    ][i] *= s[i]; 
                         dfdx[3*at + 1][i] *= s[i]; 
                         dfdx[3*at + 2][i] *= s[i];             
                    }
               }
          }
    }
    return;
}















// =====================================================================
// Public methods

template <typename T>
Gfunction_t<T>::Gfunction_t() : atom_Type_ID_t<T>(){
     xyz = nullptr;
     dfdxyz = nullptr;
     switch_factor = nullptr;
     dswitchdx = nullptr;
};

template <typename T>
Gfunction_t<T>::~Gfunction_t(){
     // clearMemo<T>(xyz);     // xyz is clean up in model member
     for(auto it=G.begin() ; it!=G.end(); it++){
          clearMemo<T>(*it);
     };

     for(auto it=G_SCALE_VAR.begin() ; it != G_SCALE_VAR.end(); it++){
          delete [] (it->second);
     };

     if(switch_factor != nullptr) delete[] switch_factor;
     clearMemo<T>(dswitchdx);
     clearMemo<T>(dfdxyz);
     clearMemo<T>(xyz);
};


template <typename T>
void Gfunction_t<T>::load_xyzfile(const char* file){
      this->load_xyz_from_file(file);
};

template <typename T>
void Gfunction_t<T>::load_paramfile(const char* file){
     if (strlen(file)>0 ){
          gparams.read_param_from_file(file);
     } else {
          load_paramfile_2h2o_default();
     };
};


template <typename T>
void Gfunction_t<T>::load_seq(const char* _seqfile){
     if ( strlen(_seqfile) >0 ){
           this->read_seq_from_file(_seqfile);
     } else {
          //  this->load_default_2h2o_3h2o_seq();
          std::cerr << " NO sequential is loaded ! " << std::endl;
     };
};


template <typename T>
void Gfunction_t<T>::load_paramfile_2h2o_default(){
     gparams.read_param_from_file("Gfunc_params_2Bv14_tuned.dat");
};


template <typename T>
void Gfunction_t<T>::load_paramfile_3h2o_default(){
     gparams.read_param_from_file("Gfunc_params_3Bv16.dat");
};


template <typename T>
void Gfunction_t<T>::load_seq_2h2o_default(){
      this->load_default_2h2o_seq();
};


template <typename T>
void Gfunction_t<T>::load_seq_3h2o_default(){
      this->load_default_3h2o_seq();
};


template <typename T>
void Gfunction_t<T>::load_scale_2h2o_default(){
     std::string _2h2o= "2h2o";
     load_resv_scales(_2h2o);
};


template <typename T>
void Gfunction_t<T>::load_scale_3h2o_default(){
     std::string _3h2o= "3h2o";
     load_resv_scales(_3h2o);
};


template <typename T>
void Gfunction_t<T>::make_G_from_files(const char* _xyzFile, const char * _paramfile, const char* _ordfile){
     
     load_xyzfile(_xyzFile);

     load_paramfile(_paramfile);

     load_seq(_ordfile);         

     make_G();
}

template <typename T>
void Gfunction_t<T>::make_G(){    
     // sort_atom_by_type_id(); // sort this->ATOMS according to type
     init_G(); 

     // idx_t * TypeStart = new idx_t[this->NTYPE];
     // idx_t * Typethis->NATOM = new idx_t[this->NTYPE];

     // {
     //      idx_t idx = 0;
     //      idx_t count = 0;
     //      for(auto it = this->this->NATOM_ONETYPE.begin(); it!= this->this->NATOM_ONETYPE.end(); it++ ){
     //           Typethis->NATOM[idx] = *it ; 
     //           TypeStart[idx] = count ;
     //           count += *it;
     //           idx ++ ;
     //      }
     // }

// for(int loop = 0; loop < 1; loop++){  // iteration test

     timers.insert_random_timer(id3, 1 , "Gf_fwd_all");
     timers.timer_start(id3);     
       
    
     size_t Cluster_Batch = this->NCLUSTER ;  // 
     // size_t id_cluster = 0 ;

// The most outer iteration is looping for every Cluster_Batch dimers/trimers
// while the most inner iteration is looping for every cluster within the Cluster_Batch
for (size_t batchstart = 0; batchstart < this->NCLUSTER; batchstart += Cluster_Batch){
      
      size_t batchlimit = std::min(batchstart+Cluster_Batch, this->NCLUSTER);

          // for each type (first type in seq)
          for(idx_t type_id = 0; type_id < this->NTYPE; type_id ++ ){

               // iterate through all paramter seq: for "H", iterate in sequence such as "HH"-"HO"-"HHO"-"HHH" depending on the seq defined
               idx_t relation_idx = 0;    // current relation writing to this index in G
               size_t offset_by_rel = 0;  // Index already finished

               for (auto rel =this->seq[type_id].begin(); rel !=this->seq[type_id].end(); rel++ ){

                    size_t np = gparams.PARAMS[*rel].nparam;  // number of parameter pairs
                    size_t nc = gparams.PARAMS[*rel].ncol;  // number of param in one line
                    T * p = (&gparams.PARAMS[*rel].dat_h[0]); // params

                    idx_t type1_id = this->seq_by_idx[type_id][relation_idx][1] ; // second type in seq


                    // if radial 
                    if( this->seq_by_idx[type_id][relation_idx].size() == 2) { 

                          // pick up an atom for type_0 (first in seq)
                          for (auto atom0_id = this->ATOMS[type_id].begin(); atom0_id != this->ATOMS[type_id].end(); atom0_id++){

                              // print the selected relationship
                              // std::cout << *rel << "  atom " << atom0_id << std::endl;
                              T** G0 = G[*atom0_id] ;   // Gfn ptr

                              // second type
                              // for atom_1 in type_1;
                              // loop only atom_1>atom_0 when type_1 == type_0  
                              for (auto atom1_id = this->ATOMS[type1_id].begin(); atom1_id != this->ATOMS[type1_id].end(); atom1_id++){        

                                   // if not same type, do it normally
                                   if ( type1_id != type_id  )  {

                                        // print the selected this->ATOMS ! 
                                        // std::cout << atom0_id << "  " << atom1_id << std::endl;

// for (size_t batchstart = 0; batchstart < this->NCLUSTER; batchstart += Cluster_Batch){
#ifdef _OPENMP
#pragma omp parallel for simd shared(atom0_id, atom1_id, batchstart, batchlimit, np, nc, G0, offset_by_rel)
#endif  
for(size_t id_cluster = batchstart; id_cluster< batchlimit; id_cluster++)
{
                                        T d = get_dist(*atom0_id, *atom1_id, id_cluster);
                                        for (size_t ip = 0; ip < np; ip ++){

                                             // print the position in G at which the result is saved to 
                                             // size_t b = offset_by_rel;
                                             // size_t c = b + ip ;
                                             // std::cout << c << " ";
                                             
                                             G0[offset_by_rel + ip ][id_cluster]  +=  get_Gradial(d, p[ip*nc + COL_RAD_RS], p[ip*nc + COL_RAD_ETA] )  ; 
                                        }

                                        // print some nice endline
                                        // std::cout << std::endl;
}
// }
                                   } else if (*atom1_id > *atom0_id) {

                                        // if same type, do it only when id1>id0
                                        // and assign the value to both G[id0] and G[id1] 

                                        // print the selected this->ATOMS ! 
                                        // std::cout << atom0_id << "  " << atom1_id << std::endl;
                                        T** G1 = G[*atom1_id];


// for (size_t batchstart = 0; batchstart < this->NCLUSTER; batchstart += Cluster_Batch){
#ifdef _OPENMP
#pragma omp parallel for simd shared(atom0_id, atom1_id, batchstart, batchlimit, np, nc, G0, G1, offset_by_rel)
#endif
for(size_t id_cluster = batchstart; id_cluster< batchlimit; id_cluster++)
{                                        
                                        T d = get_dist(*atom0_id, *atom1_id, id_cluster);
                                        for (size_t ip = 0; ip < np; ip ++){

                                             // print the position in G at which the result is saved to 
                                             // size_t b = offset_by_rel;
                                             // size_t c = b + ip ;
                                             // std::cout << c << " ";

                                             T tmp = get_Gradial(d, p[ip*nc + COL_RAD_RS], p[ip*nc + COL_RAD_ETA] ) ;
                                             G0[offset_by_rel + ip ][id_cluster]  += tmp  ; 
                                             G1[offset_by_rel + ip ][id_cluster]  += tmp ;
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
                          for (auto atom0_id = this->ATOMS[type_id].begin(); atom0_id != this->ATOMS[type_id].end() ; atom0_id++){

                              // print the selected relationship
                              // std::cout << *rel << "  atom " << atom0_id << std::endl;
                              T** G0 = G[*atom0_id] ;   // Gfn ptr

                              // the third type in seq
                              idx_t type2_id = this->seq_by_idx[type_id][relation_idx][2];

                              // for atom_1 in type_1;
                              for (auto atom1_id = this->ATOMS[type1_id].begin(); atom1_id != this->ATOMS[type1_id].end(); atom1_id++){         

                                   if (*atom0_id == *atom1_id) continue;

                                   // if type_1 == type_2, loop only atom_1 < atom_2  
                                   for (auto atom2_id = this->ATOMS[type2_id].begin(); atom2_id != this->ATOMS[type2_id].end(); atom2_id++){ 
                                        if ( *atom0_id == *atom2_id) continue; 
                                        // if (atom1_id == atom2_id) continue;

                                        // if the second and third atom are not same type, do as normal
                                        if (type1_id != type2_id) {
                                        
                                             // print the selected this->ATOMS ! 
                                             // std::cout << atom0_id << "  " << atom1_id << "  " << atom2_id << std::endl;

// for (size_t batchstart = 0; batchstart < this->NCLUSTER; batchstart += Cluster_Batch){
#ifdef _OPENMP
#pragma omp parallel for simd shared(atom0_id, atom1_id, atom2_id, batchstart, batchlimit, np, nc, G0, offset_by_rel)
#endif
for(size_t id_cluster = batchstart; id_cluster< batchlimit; id_cluster++)
{
                                             T dij = get_dist(*atom0_id, *atom1_id, id_cluster);
                                             T dik = get_dist(*atom0_id, *atom2_id, id_cluster);
                                             T djk = get_dist(*atom1_id, *atom2_id, id_cluster);

                                             for (size_t ip = 0; ip < np; ip ++){

                                                  // print the position in G at which the result is saved to 
                                                  // size_t b = offset_by_rel;
                                                  // size_t c = b + ip ;
                                                  // std::cout << c << " ";

                                                  G0[offset_by_rel + ip ][id_cluster]  +=    get_Gangular(dij, dik, djk, p[ip*nc + COL_ANG_ETA], p[ip*nc + COL_ANG_ZETA], p[ip*nc + COL_ANG_LAMBD]) ;
                                             }

                                             // print some nice endline
                                             // std::cout << std::endl;
}
// }
                                        }  else if (*atom2_id > *atom1_id) {
                                             // if second and third type are the same, do once and add twice

                                             // test the selected this->ATOMS ! 
                                             // std::cout << atom0_id << "  " << atom1_id << "  " << atom2_id << std::endl;

// for (size_t batchstart = 0; batchstart < this->NCLUSTER; batchstart += Cluster_Batch){
#ifdef _OPENMP
#pragma omp parallel for simd shared(atom0_id, atom1_id, atom2_id, batchstart, batchlimit, np, nc, G0, offset_by_rel)
#endif
for(size_t id_cluster = batchstart; id_cluster< batchlimit; id_cluster++)
{
                                             T dij = get_dist(*atom0_id, *atom1_id, id_cluster);
                                             T dik = get_dist(*atom0_id, *atom2_id, id_cluster);
                                             T djk = get_dist(*atom1_id, *atom2_id, id_cluster);

                                             for (size_t ip = 0; ip < np; ip ++){

                                                  // print the position in G at which the result is saved to 
                                                  // size_t b = offset_by_rel;
                                                  // size_t c = b + ip ;
                                                  // std::cout << c << " ";

                                                  G0[offset_by_rel + ip ][id_cluster]  +=   get_Gangular(dij, dik, djk, p[ip*nc + COL_ANG_ETA], p[ip*nc + COL_ANG_ZETA], p[ip*nc + COL_ANG_LAMBD]) ;
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

     if( this->NATOM == 6 ){
          scale_G("2h2o");
     } else if (this->NATOM == 9){
          scale_G("3h2o");
     }
     timers.timer_end(id3);

// }  // end of testing loop

     // timers.get_all_timers_info();
     // timers.get_time_collections();

};





template <typename T>
void Gfunction_t<T>::make_grd(std::vector<T**> dfdG){
     // assumes upgrd has the same dimensionality as G
     scale_G_grd(dfdG); // get the gradient with scaling factor
     
     if (dfdxyz == nullptr)
     init_mtx_in_mem(dfdxyz , (size_t)(this->NATOM *3) , this->NCLUSTER);


for(size_t loop = 0; loop < 1; loop++){

     timers.insert_random_timer(id3, 1 , "Gf_bak_all");
     timers.timer_start(id3);     

     // transpose XYZ in case it is not consistent  
     // if(!this->XYZ_ONE_ATOM_PER_COL)  this->transpose_xyz(xyz);     
      
     size_t Cluster_Batch = this->NCLUSTER ;  // 
     // size_t id_cluster = 0 ;

// The most outer iteration is looping for every Cluster_Batch dimers/trimers
// while the most inner iteration is looping for every cluster within the Cluster_Batch
for (size_t batchstart = 0; batchstart < this->NCLUSTER; batchstart += Cluster_Batch){
      
      size_t batchlimit = std::min(batchstart+Cluster_Batch, this->NCLUSTER);

          // for each type (first type in seq)
          for(idx_t type_id = 0; type_id < this->NTYPE; type_id ++ ){

               // iterate through all paramter seq: for "H", iterate in sequence such as "HH"-"HO"-"HHO"-"HHH" depending on the seq defined
               idx_t relation_idx = 0;    // current relation writing to this index in G
               size_t offset_by_rel = 0;  // Index already finished

               for (auto rel = this->seq[type_id].begin(); rel != this->seq[type_id].end(); rel++ ){

                    size_t np = gparams.PARAMS[*rel].nparam;  // number of parameter pairs
                    size_t nc = gparams.PARAMS[*rel].ncol;  // number of param in one line
                    T * p = (&gparams.PARAMS[*rel].dat_h[0]); // params

                    idx_t type1_id = this->seq_by_idx[type_id][relation_idx][1] ; // second type in seq


                    // if radial 
                    if( this->seq_by_idx[type_id][relation_idx].size() == 2) { 

                          // pick up an atom for type_0 (first in seq)
                          for (auto atom0_id = this->ATOMS[type_id].begin(); atom0_id != this->ATOMS[type_id].end(); atom0_id++){

                              // print the selected relationship
                              // std::cout << *rel << "  atom " << atom0_id << std::endl;
                              T** dfdG0 = dfdG[*atom0_id] ;   // Gfn ptr

                              // second type
                              // for atom_1 in type_1;
                              // loop only atom_1>atom_0 when type_1 == type_0  
                              for (auto atom1_id = this->ATOMS[type1_id].begin(); atom1_id != this->ATOMS[type1_id].end() ; atom1_id++){        

                                   // if not same type, do it normally
                                   if ( type1_id != type_id  )  {
                                        
                                        T** dfdG1 = dfdG[*atom1_id];
                                        // print the selected this->ATOMS ! 
                                        // std::cout << atom0_id << "  " << atom1_id << std::endl;

// for (size_t batchstart = 0; batchstart < this->NCLUSTER; batchstart += Cluster_Batch){
#ifdef _OPENMP
#pragma omp parallel for simd shared(atom0_id, atom1_id, batchstart, batchlimit, np, nc, dfdG0, offset_by_rel)
#endif  
for(size_t id_cluster = batchstart; id_cluster< batchlimit; id_cluster++)
{
                                        
                                        T d = get_dist(*atom0_id, *atom1_id, id_cluster);
                                        
                                        // get dRdX
                                        T dtmp[3]; 
                                        get_dRdX_byID(&(dtmp[0]), *atom0_id, *atom1_id, id_cluster); 

                                        for (size_t ip = 0; ip < np; ip ++){

                                             // In forward, this update G0;
                                             // In backward, this update dfdD[atom0 VS atom1]


                                             // print the position in G at which the result is saved to 
                                             // size_t b = offset_by_rel;
                                             // size_t c = b + ip ;
                                             // std::cout << c << " ";
                                             


                                             // get dfdR
                                             T dfdRtmp = dfdG0[offset_by_rel + ip ][id_cluster] * get_dGdR_rad(d, p[ip*nc + COL_RAD_RS], p[ip*nc + COL_RAD_ETA]) ;


                                             // update atom0
                                             dfdxyz[3 * *atom0_id    ][id_cluster] += dfdRtmp * dtmp[0];
                                             dfdxyz[3 * *atom0_id + 1][id_cluster] += dfdRtmp * dtmp[1];
                                             dfdxyz[3 * *atom0_id + 2][id_cluster] += dfdRtmp * dtmp[2];

                                             //update atom1
                                             dfdxyz[3 * *atom1_id    ][id_cluster] -= dfdRtmp * dtmp[0];
                                             dfdxyz[3 * *atom1_id + 1][id_cluster] -= dfdRtmp * dtmp[1];
                                             dfdxyz[3 * *atom1_id + 2][id_cluster] -= dfdRtmp * dtmp[2];
                                        }

                                        // print some nice endline
                                        // std::cout << std::endl;
}
// }
                                   } else if (*atom1_id > *atom0_id) {

                                        // if same type, do it only when id1>id0
                                        // and assign the value to both G[id0] and G[id1] 

                                        // print the selected this->ATOMS ! 
                                        // std::cout << atom0_id << "  " << atom1_id << std::endl;
                                        T** dfdG1 = dfdG[*atom1_id];


// for (size_t batchstart = 0; batchstart < this->NCLUSTER; batchstart += Cluster_Batch){
#ifdef _OPENMP
#pragma omp parallel for simd shared(atom0_id, atom1_id, batchstart, batchlimit, np, nc, dfdG0, dfdG1, offset_by_rel)
#endif
for(size_t id_cluster = batchstart; id_cluster< batchlimit; id_cluster++)
{                                       
                                        T d = get_dist(*atom0_id, *atom1_id, id_cluster);

                                        // get dRdX
                                        T dtmp[3]; 
                                        get_dRdX_byID(&(dtmp[0]),*atom0_id, *atom1_id, id_cluster);                                         
                                        for (size_t ip = 0; ip < np; ip ++){

                                             // print the position in G at which the result is saved to 
                                             // size_t b = offset_by_rel;
                                             // size_t c = b + ip ;
                                             // std::cout << c << " ";

                                             // get dfdR
                                             T dGdRtmp = get_dGdR_rad(d, p[ip*nc + COL_RAD_RS], p[ip*nc + COL_RAD_ETA]);

                                             // T dfdRtmp = ( dfdG0[offset_by_rel + ip ][id_cluster] +  dfdG1[offset_by_rel + ip ][id_cluster] ) * dGdRtmp ;

                                             T dfdRtmp = ( dfdG0[offset_by_rel + ip ][id_cluster] + dfdG1[offset_by_rel + ip ][id_cluster] ) * dGdRtmp ;

                                             // update atom0
                                             dfdxyz[3 * *atom0_id    ][id_cluster] += dfdRtmp * dtmp[0];
                                             dfdxyz[3 * *atom0_id + 1][id_cluster] += dfdRtmp * dtmp[1];
                                             dfdxyz[3 * *atom0_id + 2][id_cluster] += dfdRtmp * dtmp[2];

                                             //update atom1
                                             dfdxyz[3 * *atom1_id    ][id_cluster] -= dfdRtmp * dtmp[0];
                                             dfdxyz[3 * *atom1_id + 1][id_cluster] -= dfdRtmp * dtmp[1];
                                             dfdxyz[3 * *atom1_id + 2][id_cluster] -= dfdRtmp * dtmp[2];
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
                          for (auto atom0_id = this->ATOMS[type_id].begin(); atom0_id != this->ATOMS[type_id].end() ; atom0_id++){

                              // print the selected relationship
                              // std::cout << *rel << "  atom " << atom0_id << std::endl;
                              T** dfdG0 = dfdG[*atom0_id] ;   // Gfn ptr

                              // the third type in seq
                              idx_t type2_id = this->seq_by_idx[type_id][relation_idx][2];

                              // for atom_1 in type_1;
                              for (auto atom1_id = this->ATOMS[type1_id].begin(); atom1_id != this->ATOMS[type1_id].end(); atom1_id++){  

                                   if (*atom0_id == *atom1_id) continue;

                                   // if type_1 == type_2, loop only atom_1 < atom_2  
                                   for (auto atom2_id = this->ATOMS[type2_id].begin(); atom2_id != this->ATOMS[type2_id].end(); atom2_id++){ 
                                        if (*atom0_id == *atom2_id) continue; 
                                        // if (atom1_id == atom2_id) continue;

                                        // if the second and third atom are not same type, do as normal
                                        if (type1_id != type2_id) {
                                        
                                             // print the selected this->ATOMS ! 
                                             // std::cout << atom0_id << "  " << atom1_id << "  " << atom2_id << std::endl;

// for (size_t batchstart = 0; batchstart < this->NCLUSTER; batchstart += Cluster_Batch){
#ifdef _OPENMP
#pragma omp parallel for simd shared(atom0_id, atom1_id, atom2_id, batchstart, batchlimit, np, nc, dfdG0, offset_by_rel)
#endif
for(size_t id_cluster = batchstart; id_cluster< batchlimit; id_cluster++)
{
                                             T dij = get_dist(*atom0_id, *atom1_id, id_cluster);
                                             T dik = get_dist(*atom0_id, *atom2_id, id_cluster);
                                             T djk = get_dist(*atom1_id, *atom2_id, id_cluster);


                                             // get dRdX
                                             T dRdxtmp[3][3] ;
                                             get_dRdX_byID(dRdxtmp[0], *atom0_id, *atom1_id, id_cluster) ;
                                             get_dRdX_byID(dRdxtmp[1], *atom0_id, *atom2_id, id_cluster) ;
                                             get_dRdX_byID(dRdxtmp[2], *atom1_id, *atom2_id, id_cluster) ;


                                             for (size_t ip = 0; ip < np; ip ++){

                                                  // print the position in G at which the result is saved to 
                                                  // size_t b = offset_by_rel;
                                                  // size_t c = b + ip ;
                                                  // std::cout << c << " ";

                                                  T dfdG0tmp = dfdG0[offset_by_rel + ip ][id_cluster] ;

                                                  T dGdR01 = get_dGdR_ang_IJ(dij, dik, djk, p[ip*nc + COL_ANG_ETA], p[ip*nc + COL_ANG_ZETA], p[ip*nc + COL_ANG_LAMBD]   );

                                                  T dGdR02 = get_dGdR_ang_IJ(dik, dij, djk, p[ip*nc + COL_ANG_ETA], p[ip*nc + COL_ANG_ZETA], p[ip*nc + COL_ANG_LAMBD]   );

                                                  T dGdR12 = get_dGdR_ang_JK(dij, dik, djk, p[ip*nc + COL_ANG_ETA], p[ip*nc + COL_ANG_ZETA], p[ip*nc + COL_ANG_LAMBD]   );     


                                                  // update atom0
                                                  dfdxyz[3 * *atom0_id    ][id_cluster] += dfdG0tmp * ( dGdR01 * dRdxtmp[0][0] + dGdR02 * dRdxtmp[1][0]) ;
                                                  dfdxyz[3 * *atom0_id + 1][id_cluster] += dfdG0tmp * ( dGdR01 * dRdxtmp[0][1] + dGdR02 * dRdxtmp[1][1]) ;
                                                  dfdxyz[3 * *atom0_id + 2][id_cluster] += dfdG0tmp * ( dGdR01 * dRdxtmp[0][2] + dGdR02 * dRdxtmp[1][2]) ;                                                  

                                                  //update atom1
                                                  dfdxyz[3 * *atom1_id    ][id_cluster] += dfdG0tmp * (-dGdR01 * dRdxtmp[0][0] + dGdR12 * dRdxtmp[2][0]) ;
                                                  dfdxyz[3 * *atom1_id + 1][id_cluster] += dfdG0tmp * (-dGdR01 * dRdxtmp[0][1] + dGdR12 * dRdxtmp[2][1]) ;
                                                  dfdxyz[3 * *atom1_id + 2][id_cluster] += dfdG0tmp * (-dGdR01 * dRdxtmp[0][2] + dGdR12 * dRdxtmp[2][2]) ;                   


                                                  //update atom2
                                                  dfdxyz[3 * *atom2_id    ][id_cluster] += dfdG0tmp * (-dGdR02 * dRdxtmp[1][0] - dGdR12 * dRdxtmp[2][0]) ;
                                                  dfdxyz[3 * *atom2_id + 1][id_cluster] += dfdG0tmp * (-dGdR02 * dRdxtmp[1][1] - dGdR12 * dRdxtmp[2][1]) ;
                                                  dfdxyz[3 * *atom2_id + 2][id_cluster] += dfdG0tmp * (-dGdR02 * dRdxtmp[1][2] - dGdR12 * dRdxtmp[2][2]) ; 

                                             }

                                             // print some nice endline
                                             // std::cout << std::endl;
}
// }
                                        }  else if (*atom2_id > *atom1_id) {
                                             // if second and third type are the same, do once and add twice

                                             // test the selected this->ATOMS ! 
                                             // std::cout << atom0_id << "  " << atom1_id << "  " << atom2_id << std::endl;

// for (size_t batchstart = 0; batchstart < this->NCLUSTER; batchstart += Cluster_Batch){
#ifdef _OPENMP
#pragma omp parallel for simd shared(atom0_id, atom1_id, atom2_id, batchstart, batchlimit, np, nc, dfdG0, offset_by_rel)
#endif
for(size_t id_cluster = batchstart; id_cluster< batchlimit; id_cluster++)
{
                                             T dij = get_dist(*atom0_id, *atom1_id, id_cluster);
                                             T dik = get_dist(*atom0_id, *atom2_id, id_cluster);
                                             T djk = get_dist(*atom1_id, *atom2_id, id_cluster);

                                             // get dRdX
                                             T dRdxtmp[3][3];
                                             get_dRdX_byID(dRdxtmp[0], *atom0_id, *atom1_id, id_cluster) ;
                                             get_dRdX_byID(dRdxtmp[1], *atom0_id, *atom2_id, id_cluster) ;
                                             get_dRdX_byID(dRdxtmp[2], *atom1_id, *atom2_id, id_cluster) ;                                             

                                             for (size_t ip = 0; ip < np; ip ++){

                                                  // print the position in G at which the result is saved to 
                                                  // size_t b = offset_by_rel;
                                                  // size_t c = b + ip ;
                                                  // std::cout << c << " ";
                                                  T dfdG0tmp = dfdG0[offset_by_rel + ip ][id_cluster] ;

                                                  T dGdR01 = get_dGdR_ang_IJ(dij, dik, djk, p[ip*nc + COL_ANG_ETA], p[ip*nc + COL_ANG_ZETA], p[ip*nc + COL_ANG_LAMBD]   );

                                                  T dGdR02 = get_dGdR_ang_IJ(dik, dij, djk, p[ip*nc + COL_ANG_ETA], p[ip*nc + COL_ANG_ZETA], p[ip*nc + COL_ANG_LAMBD]   );

                                                  T dGdR12 = get_dGdR_ang_JK(dij, dik, djk, p[ip*nc + COL_ANG_ETA], p[ip*nc + COL_ANG_ZETA], p[ip*nc + COL_ANG_LAMBD]   );     


                                                  // update atom0
                                                  dfdxyz[3 * *atom0_id    ][id_cluster] += dfdG0tmp * ( dGdR01 * dRdxtmp[0][0] + dGdR02 * dRdxtmp[1][0]) ;
                                                  dfdxyz[3 * *atom0_id + 1][id_cluster] += dfdG0tmp * ( dGdR01 * dRdxtmp[0][1] + dGdR02 * dRdxtmp[1][1]) ;
                                                  dfdxyz[3 * *atom0_id + 2][id_cluster] += dfdG0tmp * ( dGdR01 * dRdxtmp[0][2] + dGdR02 * dRdxtmp[1][2]) ;                                                  

                                                  //update atom1
                                                  dfdxyz[3 * *atom1_id    ][id_cluster] += dfdG0tmp * (-dGdR01 * dRdxtmp[0][0] + dGdR12 * dRdxtmp[2][0]) ;
                                                  dfdxyz[3 * *atom1_id + 1][id_cluster] += dfdG0tmp * (-dGdR01 * dRdxtmp[0][1] + dGdR12 * dRdxtmp[2][1]) ;
                                                  dfdxyz[3 * *atom1_id + 2][id_cluster] += dfdG0tmp * (-dGdR01 * dRdxtmp[0][2] + dGdR12 * dRdxtmp[2][2]) ;                   


                                                  //update atom2
                                                  dfdxyz[3 * *atom2_id    ][id_cluster] += dfdG0tmp * (-dGdR02 * dRdxtmp[1][0] - dGdR12 * dRdxtmp[2][0]) ;
                                                  dfdxyz[3 * *atom2_id + 1][id_cluster] += dfdG0tmp * (-dGdR02 * dRdxtmp[1][1] - dGdR12 * dRdxtmp[2][1]) ;
                                                  dfdxyz[3 * *atom2_id + 2][id_cluster] += dfdG0tmp * (-dGdR02 * dRdxtmp[1][2] - dGdR12 * dRdxtmp[2][2]) ; 
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

} // end of tester loop

} // end of function make_grd




// ====================================================================
// Instanciate template with specific type
template class Gfunction_t<float>;
template class Gfunction_t<double>;







int main3l24k(int argc, char *argv[]){

     Gfunction_t<double> G;
     const char* xyzfile1 = "test1.xyz";
     const char* xyzfile = "test.xyz";
     const char* tmp = "";

     G.load_xyzfile(xyzfile1);
     G.load_paramfile("Gfunc_params_2Bv14_tuned.dat");
     G.load_seq_2h2o_default();
     G.load_scale_2h2o_default();

     G.make_G();

     G.make_G();

     std::vector<double**> dfdG;
     double** grdO = nullptr;

     std::ofstream ut;
     // ut.open("grd_set.rst");

     for(int j=0; j < 2; j++){
          for(int i = 0; i < 1; i++){
               double ** grd = nullptr;
               init_mtx_in_mem(grd, 82, 2);
               // for(int j=0; j < 82; j++) {
               //      grdO[0][i] = G.G[i][0][j]*scale;
               //      ut << std::scientific << std::setprecision(18) << grdO[0][i] << " ";
               // }
               // ut << std::endl;
               dfdG.push_back(grd);
          }

          for(int i = 0; i < 2; i++){
               double ** grd = nullptr;
               init_mtx_in_mem(grd, 84, 2);
               // for(int j=0; j < 84; j++) {
               //      grdO[0][i] = G.G[i][0][j]*scale;
               //      ut << std::scientific << std::setprecision(18) << grdO[0][i] << " ";
               // }
               // ut << std::endl;
               dfdG.push_back(grd);
          }
     }
     int checkatom = 0;
     int checkxyz = 0;

     int checkginp = 2;
     int checkpara = 75;

     checkatom = std::stoi(argv[1]);
     checkxyz = std::stoi(argv[2]);
     checkginp = std::stoi(argv[3]);
     checkpara = std::stoi(argv[4]);
     double scale = 0.00001;


     dfdG[checkginp][checkpara][0]=1.0;

     // ut.close();

     G.make_grd(dfdG);

     std::cout << "Gradient from prog is " << std::scientific << std::setprecision(18) <<G.dfdxyz[checkatom*3 + checkxyz][0]  << std::endl;



     double p_old = G.G[checkginp][checkpara][0] ;

     double dx = G.XYZ[0][checkatom*3 + checkxyz] * scale ;
     // std::cout << G.XYZ[0][checkatom*3 + checkxyz] << std::endl;
     G.XYZ[0][checkatom*3 + checkxyz] += dx ;
     // std::cout << G.XYZ[0][checkatom*3 + checkxyz] << std::endl;     
     G.make_G();

     double p_new = G.G[checkginp][checkpara][0] ;

     std::cout << "Gradient from test is " << std::scientific << std::setprecision(18) << (p_new - p_old) / dx  << std::endl;     



     for(int i =0; i< 2; i ++){
          for(int j=0; j < 18; j++){
               // std::cout <<  std::setprecision(18) << G.dfdxyz[j][i] << " | " ;
          }
          // std::cout << std::endl;
     }




     // ut.open("grd_xyz.rst");
     // for(int i = 0; i< 6; i++){
     //      for(int j = 0; j<3; j++) {
     //           ut << std::scientific << std::setprecision(18) << G.dfdxyz[0][i*3+j] << " ";
     //           G.XYZ[0][i*3+j] += G.dfdxyz[0][i*3+j] ;
     //      }
     //      ut << std::endl;
     // };
     // ut.close();




     // G.make_G();
     // ut.open("grd_test.rst");

     // for(int i = 0; i < 2; i++){
     //      // init_mtx_in_mem(grdO, 1, 82);
     //      for(int j=0; j < 82; j++) {
     //           // grdO[0][i] = G.G[i][0][j]*0.01;
     //           ut << std::scientific << std::setprecision(18) << G.G[i][0][j] << " ";
     //      }
     //      ut << std::endl;
     //      // dfdG.push_back(grdO);
     // }

     // for(int i = 2; i < 6; i++){
     //      // init_mtx_in_mem(grdO, 1, 84);
     //      for(int j=0; j < 84; j++) {
     //           // grdO[0][i] = G.G[i][0][j]*0.01;
     //           ut << std::scientific << std::setprecision(18) << G.G[i][0][j] << " ";
     //      }
     //      ut << std::endl;
     //      // dfdG.push_back(grdO);
     // }

     // ut.close();

     // G.make_G_XYZ(xyzfile1, "Gfunc_params_2Bv14.dat", tmp);


     // int i=0;
     // for(int n = 0; n< G.this->NTYPE ; n++){
     //      for(int ii =0 ; ii< G.Typethis->NATOM[n]; ii++ ){
     //           int np = G.G_param_max_size[n];
     //           for (int c = 0; c<G.this->NCLUSTER; c++){
     //                for(int p =0; p<np; p++){
     //                     ut << std::scientific << std::setprecision(18) << G.G[i][c][p] << " ";
     //                }
     //                ut << std::endl;
     //           }
     //           i++;
     //           //ut << std::endl;
     //      }
     // };

     // ut.close();

     for(auto it = dfdG.begin(); it!= dfdG.end(); it++){
          clearMemo(*it);
     }

     clearMemo(grdO);
     return 0;
}
