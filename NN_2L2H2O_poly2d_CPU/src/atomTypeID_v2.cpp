#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <queue>
#include <string>
#include <limits>
#include <algorithm>
#include <locale>
#include "atomTypeID_v2.h"
#include "utility.h"

using namespace std;


//===============================================================================
// a tester

int main1(void){
     
     
     atom_Type_ID_t<float> Model;
     const char* infile = "test.xyz";
     Model.load_xyz(infile);

     Model.read_seq_from_file("Gfn_order.dat");



     //delete _new_type;
     //delete[] colidx[0];
     //delete[] colidx;

     
     

     cout << Model.XYZ[5][3] << endl;

     cout << Model.XYZ[7][11] << endl;

     Model.sort_atom_by_type_id();


     return 0;
};


