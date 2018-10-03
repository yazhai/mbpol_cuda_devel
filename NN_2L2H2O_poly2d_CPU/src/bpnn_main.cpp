

#include "bpnn_main.h"
#include "timestamps.h"

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

  timers_t timers;
  timerid_t id;
  timers.insert_random_timer( id, 0, "NN_total");
  timers.timer_start(id);
  // NN start here
  if(!ifgrad){
    for(idx_t at = 0; at < bpnn_2h2o.NATOM; at ++){
      size_t tp_idx = bpnn_2h2o.TYPE_EACHATOM[at];
      bpnn_2h2o.nets[tp_idx].predict(
        *bpnn_2h2o.G[at], bpnn_2h2o.G_param_max_size[tp_idx], 
        bpnn_2h2o.NCLUSTER, tmp);
               
      cerr<<"Finish Network for "<<tp_idx<< " Atom "<<at<<" ."<<endl;

      for(int i = 0; i < bpnn_2h2o.NCLUSTER; i++ ){
         bpnn_2h2o.energy_[i] += tmp[i];
       }
    }

  } else {
    T * tmp_grd = nullptr;
    T** tmp_grd2= nullptr;
    std::vector<T**> dfdG;

    // run neural network on each atom and store the result in energy_ and 
    // gradient in dfdG
    for(idx_t at = 0; at < bpnn_2h2o.NATOM; at ++){
      size_t tp_idx = bpnn_2h2o.TYPE_EACHATOM[at];
      bpnn_2h2o.nets[tp_idx].predict_and_getgrad(
        *bpnn_2h2o.G[at], bpnn_2h2o.G_param_max_size[tp_idx], 
        bpnn_2h2o.NCLUSTER, tmp, tmp_grd);

      init_mtx_in_mem(tmp_grd2, bpnn_2h2o.G_param_max_size[tp_idx], 
                      bpnn_2h2o.NCLUSTER);
      copy(tmp_grd,
           tmp_grd + bpnn_2h2o.NCLUSTER * bpnn_2h2o.G_param_max_size[tp_idx],
           tmp_grd2[0] );

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
    init_mtx_in_mem(bpnn_2h2o.dfdxyz, (size_t)(bpnn_2h2o.NATOM*3), 
                    bpnn_2h2o.NCLUSTER);
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
  timers.timer_end(id);
  timers.get_all_timers_info();
  timers.get_time_collections();
                   
  delete [] tmp; 

  T energy = 0;
  for(size_t i = 0; i < bpnn_2h2o.NCLUSTER; i++ ){
    energy +=  bpnn_2h2o.energy_[i] * bpnn_2h2o.switch_factor[i];
    // TODO: replace with string defined in head file 
    // cout<<" Dimer " << i << " 's energy is " << 
    //      bpnn_2h2o.energy_[i]* bpnn_2h2o.switch_factor[i] * (EUNIT) << endl;
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
   // printf("Use defualt hdf5 for neural network:%s, which should"\
           " be found within the same folder\n", nnFile);
  }

  if( paramFile[0] == '\0'){
    if (body == 2) strncpy(paramFile, PARAM_FILE_2B, strlen(PARAM_FILE_2B));
    if (body == 3) strncpy(paramFile, PARAM_FILE_3B, strlen(PARAM_FILE_3B));
    //printf("Use defualt param file for G function:%s which should"\
           " be found within the same folder\n", paramFile);
  }

  if( orderFile[0] == '\0'){
    if (body == 2) strncpy(orderFile, PARAM_FILE_2B, strlen(PARAM_FILE_2B));
    if (body == 3) strncpy(orderFile, PARAM_FILE_3B, strlen(PARAM_FILE_3B));
   // printf("Use defualt order file for G function:%s which should"\
            " be found within the same folder\n", orderFile);
  }

  if( scaleFile[0] == '\0'){
    if (body == 2) strncpy(scaleFile, PARAM_FILE_2B, strlen(PARAM_FILE_2B));
    if (body == 3) strncpy(scaleFile, PARAM_FILE_3B, strlen(PARAM_FILE_3B));
    //printf("Use defualt scale file for G function:%s which should"\
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
  int calGradient = 0;
  int body = 2;
  FILE * fileTester;

  if(argc <2 ){
     printf("Note enough parameters. \n Please use '-h' for more information.");
     return 0; 
  } 
  
  
  strcpy(inputFile,argv[1]);
     
  // Parse flags
  while(1){
    optPlaceholder = getopt_long(argc, argv, "i:b:g:p:o:s:n:h", 
                                 input_options, nullptr);

    if (optPlaceholder == -1) break;

    switch(optPlaceholder){
      case 'i':
        memset(&inputFile[0], 0, sizeof(inputFile) ) ;
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
