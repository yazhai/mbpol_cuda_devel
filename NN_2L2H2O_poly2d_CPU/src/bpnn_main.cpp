

#include "bpnn_main.h"
#include <iomanip>
#include <getopt.h>
#include <string.h>

using namespace std;
using namespace MBbpnnPlugin;


template<typename T>
bpnn_t<T>::bpnn_t(): Gfunction_t<T>(), allNN_t<T>(), energy_(nullptr){} ;

template<typename T>
bpnn_t<T>::bpnn_t(string tag): Gfunction_t<T>(), allNN_t<T>(), energy_(nullptr){
     if (tag == "2h2o_default") {

          cerr << " === Initialize default 2B H2O model ... === " << endl;          

          this->insert_atom("O");
          this->insert_atom("H");
          this->insert_atom("H");
          this->insert_atom("O");
          this->insert_atom("H");
          this->insert_atom("H");

          this->load_seq_2h2o_default();
          this->load_paramfile_2h2o_default();
          this->load_scale_2h2o_default();
          this->init_allNNs(2, nnFile, CHECKCHAR2);
          
          cerr << " === Model initialized successfully ! === " << endl;
     } else if (tag == "3h2o_default") {
          cerr << " === Initialize default 3B H2O model ... === " << endl;          

          this->insert_atom("O");
          this->insert_atom("H");
          this->insert_atom("H");
          this->insert_atom("O");
          this->insert_atom("H");
          this->insert_atom("H");
          this->insert_atom("O");
          this->insert_atom("H");
          this->insert_atom("H");          

          this->load_seq_3h2o_default();
          this->load_paramfile_3h2o_default();
          this->load_scale_3h2o_default();

          this->init_allNNs(2, nnFile, CHECKCHAR2);
          
          cerr << " === Model initialized successfully ! === " << endl;

     };
} ;

template<typename T>
bpnn_t<T>::~bpnn_t(){
     if (energy_ != nullptr ) delete [] energy_ ; 
};





template<typename T>
T MBbpnnPlugin::get_eng_2h2o(size_t nd, std::vector<T>xyz1, std::vector<T>xyz2, std::vector<std::string> atoms1, std::vector<std::string> atoms2){ 

     string tag = "2h2o_default";
     static bpnn_t<T> bpnn_2h2o(tag);

     std::vector<T> allxyz;

     size_t a1 = (size_t) ( xyz1.size() / (3 * nd) ); 
     size_t a2 = (size_t) ( xyz2.size() / (3 * nd) ); 


     if ( ! atoms1.empty() ) {
          if (a1 != atoms1.size() ) cerr << " Molecule 1 type inconsistent! " << endl;
          if (a2 != atoms2.size() ) cerr << " Molecule 2 type inconsistent! " << endl;
          atoms1.insert(atoms1.end(), atoms2.begin(), atoms2.end() );
     }

     auto it1 = xyz1.begin();
     auto it2 = xyz2.begin();

     for(size_t i =0; i< nd; i++){
          for(int j = 0; j < a1; j++){
               allxyz.push_back(*it1++) ;
               allxyz.push_back(*it1++) ;
               allxyz.push_back(*it1++) ;
          }
          for(int j = 0; j < a2; j++){
               allxyz.push_back(*it2++) ;
               allxyz.push_back(*it2++) ;
               allxyz.push_back(*it2++) ;
          }               
     }

     bpnn_2h2o.load_xyz_and_type_from_vectors(nd, allxyz, atoms1) ; 

     bpnn_2h2o.make_G();
     bpnn_2h2o.cal_switch(2);

     if (bpnn_2h2o.energy_ != nullptr) delete[] bpnn_2h2o.energy_ ;

     bpnn_2h2o.energy_ = new T [bpnn_2h2o.NCLUSTER]();
     T * tmp = new T [bpnn_2h2o.NCLUSTER]() ;

     for(idx_t at = 0; at < bpnn_2h2o.NATOM; at ++){

          size_t tp_idx = bpnn_2h2o.TYPE_EACHATOM[at];

          bpnn_2h2o.nets[tp_idx].predict(*bpnn_2h2o.G[at], bpnn_2h2o.G_param_max_size[tp_idx], bpnn_2h2o.NCLUSTER, tmp );

          for (int i = 0; i < bpnn_2h2o.NCLUSTER; i++ ){
               bpnn_2h2o.energy_[i] += tmp[i];
          }
          
     }

     delete [] tmp; 

     T energy = 0;

     for (size_t i = 0; i < bpnn_2h2o.NCLUSTER; i++ ){
          energy +=  bpnn_2h2o.energy_[i] * bpnn_2h2o.switch_factor[i];
          // cout << " Dimer " << i << " 's energy is " << bpnn_2h2o.energy_[i]* bpnn_2h2o.switch_factor[i] * (EUNIT) << endl;
     }
         
     // cout << " Total energy is "<< energy << endl;                         
     return energy*(EUNIT);
}


template<typename T>
T MBbpnnPlugin::get_eng_2h2o(size_t nd, std::vector<T>xyz1, std::vector<T>xyz2, std::vector<T> & grad1, std::vector<T>& grad2, std::vector<std::string> atoms1, std::vector<std::string> atoms2 ){ 

     string tag = "2h2o_default";
     static bpnn_t<T> bpnn_2h2o(tag);

     std::vector<T> allxyz;

     size_t a1 = (size_t) ( xyz1.size() / (3 * nd) ); 
     size_t a2 = (size_t) ( xyz2.size() / (3 * nd) ); 


     if ( ! atoms1.empty() ) {
          if (a1 != atoms1.size() ) cerr << " Molecule 1 type inconsistent! " << endl;
          if (a2 != atoms2.size() ) cerr << " Molecule 2 type inconsistent! " << endl;
          atoms1.insert(atoms1.end(), atoms2.begin(), atoms2.end() );
     }

     auto it1 = xyz1.begin();
     auto it2 = xyz2.begin();

     for(size_t i =0; i< nd; i++){
          for(int j = 0; j < a1; j++){
               allxyz.push_back(*it1++) ;
               allxyz.push_back(*it1++) ;
               allxyz.push_back(*it1++) ;
          }
          for(int j = 0; j < a2; j++){
               allxyz.push_back(*it2++) ;
               allxyz.push_back(*it2++) ;
               allxyz.push_back(*it2++) ;
          }               
     }

     bpnn_2h2o.load_xyz_and_type_from_vectors(nd, allxyz, atoms1) ; 

     bpnn_2h2o.make_G();
     bpnn_2h2o.cal_switch_with_grad(2);



     if (bpnn_2h2o.energy_ != nullptr) delete[] bpnn_2h2o.energy_ ;

     bpnn_2h2o.energy_ = new T [bpnn_2h2o.NCLUSTER]();
     T * tmp = new T [bpnn_2h2o.NCLUSTER]() ;

     T * tmp_grd = nullptr;
     T** tmp_grd2= nullptr;

     std::vector<T**> dfdG;

     for(idx_t at = 0; at < bpnn_2h2o.NATOM; at ++){

          size_t tp_idx = bpnn_2h2o.TYPE_EACHATOM[at];

          bpnn_2h2o.nets[tp_idx].predict_and_getgrad(*bpnn_2h2o.G[at], bpnn_2h2o.G_param_max_size[tp_idx], bpnn_2h2o.NCLUSTER, tmp, tmp_grd );

          for (int i = 0; i < bpnn_2h2o.NCLUSTER; i++ ){
               bpnn_2h2o.energy_[i] += tmp[i];
          }

          init_mtx_in_mem(tmp_grd2, bpnn_2h2o.G_param_max_size[tp_idx], nd) ;
          copy(tmp_grd, tmp_grd + nd * bpnn_2h2o.G_param_max_size[tp_idx],tmp_grd2[0] );

          if (tmp_grd != nullptr){
               delete [] tmp_grd;
               tmp_grd = nullptr;
          }
          dfdG.push_back(tmp_grd2);
          // tmp_grd
     }
     delete [] tmp; 
     if (tmp_grd != nullptr) delete[] tmp_grd;

     // save partially gradient       
     init_mtx_in_mem(bpnn_2h2o.dfdxyz , (size_t)(bpnn_2h2o.NATOM *3) , bpnn_2h2o.NCLUSTER);
     bpnn_2h2o.make_grd(dfdG);

     for (auto it= dfdG.begin(); it!= dfdG.end(); it++){
          clearMemo(*it);
     }

     // another part of gradient comes from switching function
     bpnn_2h2o.scale_dfdx_with_switch(2, bpnn_2h2o.energy_);

     for(size_t j=0; j < bpnn_2h2o.NCLUSTER; j++){
          for(size_t i =0; i< bpnn_2h2o.NATOM*3; i++){
                    cout << bpnn_2h2o.dfdxyz[i][j] * (EUNIT) << "   ";
                    // bpnn_2h2o.dfdxyz[i][j] *= (EUNIT) ; 
               }
          cout << endl;
     }


     T* grd = bpnn_2h2o.dfdxyz[0];
     for(size_t i =0; i< nd; i++){
          for(int j = 0; j < a1; j++){
               grad1.push_back(*grd++) ;
               grad1.push_back(*grd++) ;
               grad1.push_back(*grd++) ;
          }
          for(int j = 0; j < a2; j++){
               grad2.push_back(*grd++) ;
               grad2.push_back(*grd++) ;
               grad2.push_back(*grd++) ;
          }             
     }


     T energy = 0;

     for (size_t i = 0; i < bpnn_2h2o.NCLUSTER; i++ ){
          energy +=  bpnn_2h2o.energy_[i] * bpnn_2h2o.switch_factor[i];
          // cout << " Dimer " << i << " 's energy is " << bpnn_2h2o.energy_[i]*  bpnn_2h2o.switch_factor[i] * (EUNIT) << endl;
     }
         


     return energy*(EUNIT);
}



// real get_eng_2h2o fired in main
template<typename T>
T MBbpnnPlugin::get_eng_2h2o(const char* xyzfile, bool ifgrad ){ 

  static bpnn_t<T> bpnn_2h2o;
  bpnn_2h2o.load_xyzfile(xyzfile) ; 

  bpnn_2h2o.load_seq_2h2o_default();
  bpnn_2h2o.load_paramfile_2h2o_default();
  bpnn_2h2o.load_scale_2h2o_default();
  bpnn_2h2o.init_allNNs(2, nnFile, CHECKCHAR2);

  bpnn_2h2o.make_G();
  if( ifgrad ){   
    bpnn_2h2o.cal_switch_with_grad(2);
  } else {
    bpnn_2h2o.cal_switch(2);
  }

  if (bpnn_2h2o.energy_ != nullptr) delete[] bpnn_2h2o.energy_ ;

  bpnn_2h2o.energy_ = new T [bpnn_2h2o.NCLUSTER]();
  T * tmp = new T [bpnn_2h2o.NCLUSTER]() ;

  if(!ifgrad){
    for(idx_t at = 0; at < bpnn_2h2o.NATOM; at ++){
      size_t tp_idx = bpnn_2h2o.TYPE_EACHATOM[at];
      bpnn_2h2o.nets[tp_idx].predict(*bpnn_2h2o.G[at], bpnn_2h2o.G_param_max_size[tp_idx], bpnn_2h2o.NCLUSTER, tmp );
               
      cout<<"Finish Network for "<<tp_idx<< " Atom "<<at<<" ."<<endl;

      for(int i = 0; i < bpnn_2h2o.NCLUSTER; i++ ){
         bpnn_2h2o.energy_[i] += tmp[i];
       }
    }

  } else {
    T * tmp_grd = nullptr;
    T** tmp_grd2= nullptr;
    std::vector<T**> dfdG;

    // run neural network on each atom and store the result in energy_ and gradient in dfdG
    for(idx_t at = 0; at < bpnn_2h2o.NATOM; at ++){
      size_t tp_idx = bpnn_2h2o.TYPE_EACHATOM[at];
      bpnn_2h2o.nets[tp_idx].predict_and_getgrad(*bpnn_2h2o.G[at], bpnn_2h2o.G_param_max_size[tp_idx], bpnn_2h2o.NCLUSTER, tmp, tmp_grd);

      init_mtx_in_mem(tmp_grd2, bpnn_2h2o.G_param_max_size[tp_idx], bpnn_2h2o.NCLUSTER) ;
      copy(tmp_grd, tmp_grd + bpnn_2h2o.NCLUSTER * bpnn_2h2o.G_param_max_size[tp_idx],tmp_grd2[0] );

      if (tmp_grd != nullptr){
        delete [] tmp_grd;
        tmp_grd = nullptr;
      }
      dfdG.push_back(tmp_grd2);
         // tmp_grd

      for (int i = 0; i < bpnn_2h2o.NCLUSTER; i++ ){
         bpnn_2h2o.energy_[i] += tmp[i];
      }
      // return 0;
    }

    // calculate the gradient introduced by G function 
    init_mtx_in_mem(bpnn_2h2o.dfdxyz , (size_t)(bpnn_2h2o.NATOM *3) , bpnn_2h2o.NCLUSTER);
    bpnn_2h2o.make_grd(dfdG);

    for(auto it= dfdG.begin(); it!= dfdG.end(); it++){
      clearMemo(*it);
    }

    // another part of gradient comes from switching function
    bpnn_2h2o.scale_dfdx_with_switch(2, bpnn_2h2o.energy_);

    for(size_t j=0; j < bpnn_2h2o.NCLUSTER; j++){
      for(size_t i =0; i< bpnn_2h2o.NATOM*3; i++){
        cout << bpnn_2h2o.dfdxyz[i][j] * (EUNIT) << "   ";
        // bpnn_2h2o.dfdxyz[i][j] *= (EUNIT) ; 
      }
      cout << endl;
    }

    cout << " gradient is done, but not returned " << endl;
  }
                   
  delete [] tmp; 

  T energy = 0;
  for(size_t i = 0; i < bpnn_2h2o.NCLUSTER; i++ ){
    energy +=  bpnn_2h2o.energy_[i] * bpnn_2h2o.switch_factor[i];
    // TODO: replace with string defined in head file 
    cout << " Dimer " << i << " 's energy is " << bpnn_2h2o.energy_[i]* bpnn_2h2o.switch_factor[i] * (EUNIT) << endl;
  }

  return energy*(EUNIT);
}




template<typename T>
T MBbpnnPlugin::get_eng_3h2o(const char* xyzfile, bool ifgrad ){ 

     static bpnn_t<T> bpnn_3h2o;
     bpnn_3h2o.load_xyzfile(xyzfile) ; 

     bpnn_3h2o.load_seq_3h2o_default();
     bpnn_3h2o.load_paramfile_3h2o_default();
     bpnn_3h2o.load_scale_3h2o_default();
     bpnn_3h2o.init_allNNs(2, nnFile, CHECKCHAR2);

     bpnn_3h2o.make_G();
     if (ifgrad) {   
          bpnn_3h2o.cal_switch_with_grad(3);
     } else {
          bpnn_3h2o.cal_switch(3);
     }

     if (bpnn_3h2o.energy_ != nullptr) delete[] bpnn_3h2o.energy_ ;

     bpnn_3h2o.energy_ = new T [bpnn_3h2o.NCLUSTER]();


     T * tmp = new T [bpnn_3h2o.NCLUSTER]() ;

     if(!ifgrad){
          for(idx_t at = 0; at < bpnn_3h2o.NATOM; at ++){
               size_t tp_idx = bpnn_3h2o.TYPE_EACHATOM[at];
               bpnn_3h2o.nets[tp_idx].predict(*bpnn_3h2o.G[at], bpnn_3h2o.G_param_max_size[tp_idx], bpnn_3h2o.NCLUSTER, tmp );

               for (int i = 0; i < bpnn_3h2o.NCLUSTER; i++ ){
                    bpnn_3h2o.energy_[i] += tmp[i];
               }
          }

     } else {

          T * tmp_grd = nullptr;
          T** tmp_grd2= nullptr;

          std::vector<T**> dfdG;


          for(idx_t at = 0; at < bpnn_3h2o.NATOM; at ++){

               size_t tp_idx = bpnn_3h2o.TYPE_EACHATOM[at];
               
               bpnn_3h2o.nets[tp_idx].predict_and_getgrad(*bpnn_3h2o.G[at], bpnn_3h2o.G_param_max_size[tp_idx], bpnn_3h2o.NCLUSTER, tmp, tmp_grd);

               init_mtx_in_mem(tmp_grd2, bpnn_3h2o.G_param_max_size[tp_idx], bpnn_3h2o.NCLUSTER) ;
               copy(tmp_grd, tmp_grd + bpnn_3h2o.NCLUSTER * bpnn_3h2o.G_param_max_size[tp_idx],tmp_grd2[0] );

               if (tmp_grd != nullptr){
                    delete [] tmp_grd;
                    tmp_grd = nullptr;
               }
               dfdG.push_back(tmp_grd2);
               // tmp_grd

               for (int i = 0; i < bpnn_3h2o.NCLUSTER; i++ ){
                    bpnn_3h2o.energy_[i] += tmp[i];
               }               
          }


          

          // save partially gradient       
          init_mtx_in_mem(bpnn_3h2o.dfdxyz , (size_t)(bpnn_3h2o.NATOM *3) , bpnn_3h2o.NCLUSTER);
          bpnn_3h2o.make_grd(dfdG);

          for (auto it= dfdG.begin(); it!= dfdG.end(); it++){
               clearMemo(*it);
          }

          // another part of gradient comes from switching function
          bpnn_3h2o.scale_dfdx_with_switch(3, bpnn_3h2o.energy_);

          for(size_t j=0; j < bpnn_3h2o.NCLUSTER; j++){
          for(size_t i =0; i< bpnn_3h2o.NATOM*3; i++){
               
                    cout << bpnn_3h2o.dfdxyz[i][j] * 1 << "   " ;
               }
               cout << endl;
          }

          cout << " gradient is done, but not returned " << endl;


     }

     delete [] tmp; 

     T energy = 0;
     for (size_t i = 0; i < bpnn_3h2o.NCLUSTER; i++ ){
          energy +=  bpnn_3h2o.energy_[i] * bpnn_3h2o.switch_factor[i];
          // cout << " Dimer " << i << " 's energy is " << bpnn_3h2o.energy_[i]* bpnn_3h2o.switch_factor[i] * (EUNIT) << endl;
          //
          // TODO: replace with fixed string defined in head file 

          cout << " Dimer " << i << " 's energy is " << bpnn_3h2o.energy_[i]* bpnn_3h2o.switch_factor[i] *1 << endl;

     }





     return energy*(EUNIT);
}












template class bpnn_t<float>;
template class bpnn_t<double>;

template float MBbpnnPlugin::get_eng_2h2o<float>(size_t nd, std::vector<float>xyz1, std::vector<float>xyz2, std::vector<std::string> atoms1, std::vector<std::string> atoms2);

template double MBbpnnPlugin::get_eng_2h2o<double>(size_t nd, std::vector<double>xyz1, std::vector<double>xyz2, std::vector<std::string> atoms1, std::vector<std::string> atoms2);



template float MBbpnnPlugin::get_eng_2h2o<float>(size_t nd, std::vector<float>xyz1, std::vector<float>xyz2, std::vector<float>& grad1, std::vector<float>& grad2,std::vector<std::string> atoms1, std::vector<std::string> atoms2);

template double MBbpnnPlugin::get_eng_2h2o<double>(size_t nd, std::vector<double>xyz1, std::vector<double>xyz2, 
std::vector<double>& grad1, std::vector<double>& grad2,std::vector<std::string> atoms1, std::vector<std::string> atoms2);




template float MBbpnnPlugin::get_eng_2h2o<float>(const char* infile, bool ifgrad);

template double MBbpnnPlugin::get_eng_2h2o<double>(const char* infile, bool ifgrad);

template float MBbpnnPlugin::get_eng_3h2o<float>(const char* infile, bool ifgrad);

template double MBbpnnPlugin::get_eng_3h2o<double>(const char* infile, bool ifgrad);









int main_bak(int argc, const char* argv[]){

     cout << " usage: THIS.EXE 2|3 in.xyz if_grad[0|1]" << endl;

     if(argc >2 )  {
          bool ifgrad = false ;
          if(argc == 4 ) {
               if (atoi(argv[3]) == 1) ifgrad = true;
          }

          if ( atoi(argv[1]) == 2 ){
               get_eng_2h2o<double>(argv[2], ifgrad);
          } else if ( atoi(argv[1]) == 3 ) {
               // get_eng_3h2o(argv[2], ifgrad);
          }
     // } else {


     vector<double> xyz1, xyz2, grad1, grad2;
     double k = 0.001;
     vector<string> atoms;
     atoms.push_back("O");
     atoms.push_back("H");
     atoms.push_back("H");

     double X1[] = {     
 6.63713761738e-02 , 0.00000000000e+00 , 2.77747931775e-03,
-5.58841194785e-01 , 9.44755044560e-03 , 7.46468602271e-01,
-4.94520768565e-01 ,-9.44755044560e-03 ,-7.90549217078e-01
     };

     double X2[] = {     
-5.53091751468e-02 , 3.57455079421e-02  , 2.91048792936e+00,
 7.01247528930e-02 ,-7.30912395468e-01  , 3.49406182382e+00,
 8.07672042691e-01 , 1.63605212803e-01  , 2.48273588087e+00
     };


     double X3[] = {
 6.64009230786e-02 , 0.00000000000e+00 , 1.94714685775e-03,
-5.49462991968e-01 ,-8.87993450068e-04 , 7.53457194072e-01,
-5.04367902227e-01 , 8.87993450068e-04 ,-7.84359829444e-01
     };

     double X4[] = {
-5.76867978356e-02 ,-4.88696559854e-03  , 4.23354333315e+00,
 7.66626879677e-02 , 1.05368656403e-01  , 5.18949857004e+00,
 8.38868707197e-01 ,-2.78089616304e-02  , 3.85975287053e+00
     };

     double dX1[9];
     double dX2[9];



     for(int i=0; i <9 ; i++){
          xyz1.push_back(X1[i]);
          xyz2.push_back(X2[i]);
     }

     size_t n = 1 ;
     double e1, e2;
     e1 = get_eng_2h2o(n, xyz1, xyz2) ;
     e2 = get_eng_2h2o(n, xyz1, xyz2, grad1, grad2) ;


     // for( int i=0; i <9; i++){
     //      {
     //           vector<double> x1, x2;
     //           x1 = xyz1;
     //           x2 = xyz2;
     //           double d = x1[i] * k;
     //           x1[i] += d;
     //           double e = get_eng_2h2o(n, x1, x2) ; 
     //           double de = e - e1;
     //           dX1[i] = de / d; 
     //      };

     //      {
     //           vector<double> x1, x2;
     //           x1 = xyz1;
     //           x2 = xyz2;
     //           double d = x2[i] * k;
     //           x2[i] += d;
     //           double e = get_eng_2h2o(n, x1, x2) ; 
     //           double de = e - e1;
     //           dX2[i] = de / d; 
     //      };
     // }

     cout << " Energy is " << e1 << " and " << e2 << " wo/w grad respectively, and reference energy is: 2.717692114 " << endl;


     // cout << " Cal gradient(above) VS gradient by input(mid) VS reference graident(bot) is: " << endl;
     // for(auto it=grad1.begin(); it!= grad1.end(); it++){
     //      cout << setw(12) <<*it << " ";
     // }
     // for(auto it=grad2.begin(); it!= grad2.end(); it++){
     //      cout << setw(12) <<*it << " ";
     // }     
     // cout << endl;
     // for(int i=0; i < 9; i++){
     //      cout << setw(12) << dX1[i] << " ";
     // }     
     // for(int i=0; i < 9; i++){
     //      cout << setw(12) << dX2[i] << " ";
     // }          
     // cout << endl;
     // cout << "-0.724757268 -0.130047727 -3.143973336 -1.617402502 -0.718120693 -5.651154603 -0.075997713  0.355257790 -1.548671998  0.971052895  1.943847983  6.554913834  0.592645772 -1.069982368  0.737319658  0.854458816 -0.380954985  3.051566444" << endl;




     // xyz1.clear();
     // xyz2.clear();
     // grad1.clear();
     // grad2.clear();

     for(int i=0; i <9 ; i++){
          xyz1.push_back(X3[i]);
          xyz2.push_back(X4[i]);
     }
     n = 2;
     e1 = get_eng_2h2o(n, xyz1, xyz2) ;
     e2 = get_eng_2h2o(n, xyz1, xyz2, grad1, grad2) ;

     // for( int i=0; i <9; i++){
     //      {
     //           vector<double> x1, x2;
     //           x1 = xyz1;
     //           x2 = xyz2;
     //           double d = x1[i] * k;
     //           x1[i] += d;
     //           double e = get_eng_2h2o(n, x1, x2) ; 
     //           double de = e - e1;
     //           dX1[i] = de / d; 
     //      };

     //      {
     //           vector<double> x1, x2;
     //           x1 = xyz1;
     //           x2 = xyz2;
     //           double d = x2[i] * k;
     //           x2[i] += d;
     //           double e = get_eng_2h2o(n, x1, x2) ; 
     //           double de = e - e1;
     //           dX2[i] = de / d; 
     //      };
     // }

     cout << " Energy is " << e1 << " and " << e2 << " wo/w grad respectively, and reference energy is: 0.090248572 " << endl;


     // cout << " Cal gradient(above) VS gradient by input(mid) VS reference graident(bot) is: " << endl;
     // for(auto it=grad1.begin(); it!= grad1.end(); it++){
     //      cout << setw(12) <<*it << " ";
     // }
     // for(auto it=grad2.begin(); it!= grad2.end(); it++){
     //      cout << setw(12) <<*it << " ";
     // }     
     // cout << endl;
     // for(int i=0; i < 9; i++){
     //      cout << setw(12) << dX1[i] << " ";
     // }     
     // for(int i=0; i < 9; i++){
     //      cout << setw(12) << dX2[i] << " ";
     // }          
     // cout << endl;
     // cout << " 0.138880070 -0.011111458  0.084297440 -0.202194333  0.033384194 -0.291813818 -0.043120857 -0.007868340  0.097843290  0.349070577 -0.025231151  0.079072851 -0.119690823  0.008111216 -0.165551883 -0.122944634  0.002715540  0.196152120" << endl;


     }
     return 0;
}

static const struct option input_options[] = {
  {"input", required_argument, 0, 'i'},
  {"body", required_argument, 0, 'b'},
  {"gradient", required_argument, 0, 'g'},
  {"param",  required_argument, 0, 'p'},
  {"order",  required_argument, 0, 'o'},
  {"scale",  required_argument, 0, 's'},
  {"nn",    required_argument, 0, 'n'},
  {"help", no_argument, 0, 'h'},
  {0, 0, 0, 0}
};


/*
 * Function Name: checkFile()
 * Function Prototype: int checkFile(char* buffer, char* optarg)
 * Description: Copy file name onto buffer and check it can be opened or not
 * Parameters: 
 *  buffer - placeholder to store the string, should be big enough
 *  optarg - char[] storing file path
 * Side Effects: overwrite content of buffer
 * Error Conditions: File cannot be opened 
 * Return Value: 0 if success, otherwise -1
 */
int checkFile(char* buffer, char* optarg){
  strncpy(buffer, optarg, strlen(optarg));

  FILE * fileTester = fopen (buffer,"r");
  if( fileTester==NULL ){
    printf(OPEN_FILE_ERROR, buffer);
    return -1;
  }else{
    fclose(fileTester);
  }
  return 0;
}

/*
 * Function Name: checkDefaultFile()
 * Function Prototype: int checkDefaultFile(int body, 
 *                                        char[] paramFile, 
 *                                        char[] orderFile,
 *                                        char[] scaleFile, 
 *                                        char[] nnFile)
 * Description: fall back to default input files if not specify
 * Parameters:
 *  body - number of body, <2|3>
 *  paramFile - char[] for parameter file for G function
 *  orderFile - char[] for order file for G function
 *  scaleFile - char[] for scale ile for G function
 *  nnFile - char[] for parameter file for neural network
 * Side Effects: update char[] if not empty
 * Error Conditions: None 
 * Return Value: None 
 */
void checkDefaultFile(int body, char paramFile[], char orderFile[], 
                      char scaleFile[], char nnFile[]){

  if ( nnFile[0] == '\0') {
    // use default hdf5 file, defined in ../inc/bpnn_main.h
    if (body == 2) strncpy(nnFile, INFILE_2B, strlen(INFILE_2B));
    if (body == 3) strncpy(nnFile, INFILE_3B, strlen(INFILE_3B));
    printf("Use defualt hdf5 for neural network:%s, which should"\
           " be found within the same folder\n", nnFile);
  }

  if( paramFile[0] == '\0'){
    if (body == 2) strncpy(paramFile, PARAM_FILE_2B, strlen(PARAM_FILE_2B));
    if (body == 3) strncpy(paramFile, PARAM_FILE_3B, strlen(PARAM_FILE_3B));
    printf("Use defualt param file for G function:%s which should"\
           " be found within the same folder\n", paramFile);
  }

  if( orderFile[0] == '\0'){
    if (body == 2) strncpy(orderFile, PARAM_FILE_2B, strlen(PARAM_FILE_2B));
    if (body == 3) strncpy(orderFile, PARAM_FILE_3B, strlen(PARAM_FILE_3B));
    printf("Use defualt order file for G function:%s which should"\
            " be found within the same folder\n", orderFile);
  }

  if( scaleFile[0] == '\0'){
    if (body == 2) strncpy(scaleFile, PARAM_FILE_2B, strlen(PARAM_FILE_2B));
    if (body == 3) strncpy(scaleFile, PARAM_FILE_3B, strlen(PARAM_FILE_3B));
    printf("Use defualt scale file for G function:%s which should"\
           " be found within the same folder\n", scaleFile);
  }


}

char inputFile[BUFSIZ]; // moduar
char paramFile[BUFSIZ]; // parameter file for G function
char orderFile[BUFSIZ]; // order file for G function
char scaleFile[BUFSIZ]; // scale file for G function
char nnFile[BUFSIZ];    // parameter file for Neural Network

int main(int argc, char* argv[]){
  int optPlaceholder = -1;
  int calGradient = -1;
  int body = -1;
  FILE * fileTester;

     
  // Parse flags
  while(1){
    optPlaceholder = getopt_long(argc, argv, "i:b:g:p:o:s:n:h", 
                                 input_options, nullptr);

    if (optPlaceholder == -1) break;

    switch(optPlaceholder){
      case 'i':
        if( checkFile(inputFile, optarg) == -1) return EXIT_FAILURE;
        printf("input file: %s\n", inputFile);
        break;

      case 'p':
        if( checkFile(paramFile, optarg) == -1) return EXIT_FAILURE;
        printf("param file for G function: %s\n", paramFile);
        break;

      case 'o':
        if( checkFile(orderFile, optarg) == -1) return EXIT_FAILURE;
        printf("order file for G function: %s\n", orderFile);
        break;

      case 'n':
        if( checkFile(nnFile, optarg) == -1) return EXIT_FAILURE;
        printf("parameter file for Neural Network %s\n", nnFile);
        break;

      case 'g':
        if( strncmp(optarg, "1", 1)!=0 && strncmp(optarg, "0", 1)!=0){
          printf("-g, --gradient <0|1>\n");
          return EXIT_FAILURE;
        }
        printf("calculate gradient: %s\n",
                strncmp(optarg, "1", 1) == 0 ? "True": "False");
        calGradient = atoi(optarg);
        break;

      case 'b':
        body = atoi(optarg);
        if(body != 2 && body != 3){
          printf("-b, --body <2|3>\n");
          return EXIT_FAILURE;
        }
        break;
        
      case 'h':
        printf(STR_USAGE_LONG, argv[0]);
        return EXIT_SUCCESS;

      case '?':
        printf(STR_USAGE_SHORT, argv[0], argv[0], argv[0]);
        return EXIT_FAILURE;
        break;

      default:
        break;
    }
  }

  if(inputFile[0] == '\0'){
    printf("Missing files, abort\n");
    printf(STR_USAGE_SHORT, argv[0], argv[0], argv[0]);
    return EXIT_FAILURE;
  }

  if(calGradient == -1){
    printf("Missing -g flag, abort\n");
    printf(STR_USAGE_SHORT, argv[0], argv[0], argv[0]);
    return EXIT_FAILURE;
  }

  if(body == -1){
    printf("Missing -b flag, abort\n");
    printf(STR_USAGE_SHORT, argv[0], argv[0], argv[0]);
    return EXIT_FAILURE;

  } else {
    checkDefaultFile(body, paramFile, orderFile, scaleFile, nnFile);

  }


  // Fire energy calculation
  if(body == 2)
    get_eng_2h2o<double>(inputFile, calGradient);
  else if(body == 3)
    get_eng_3h2o<double>(inputFile, calGradient);

  return 0;
}
