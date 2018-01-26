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

using namespace std;


//===============================================================================
// a tester

int mainTD(void){
     
     
     atom_Type_ID_t2<float> Model;
     
     Model.insert_atom("O");   
     Model.insert_atom("H");           
     Model.insert_atom("H"); 
     Model.insert_atom("O");
     Model.insert_atom("H"); 
     Model.insert_atom("H"); 



     Model.read_seq_from_file("Gfn_order.dat");


     const char* infile = "test.xyz";
     //delete _new_type;
     //delete[] colidx[0];
     //delete[] colidx;

     Model.load_xyz(infile);
    for(auto it=Model.TYPE_INDEX.begin(); it!= Model.TYPE_INDEX.end(); it++){ 
     	cout<<*it<<endl;
    } 
	cout<<endl;
    for(auto it=Model.TYPE_EACHATOM.begin(); it!= Model.TYPE_EACHATOM.end(); it++){ 
     cout<<*it<<endl;
    }
	cout<<endl;
	for(auto it=Model.NATOM_ONETYPE.begin(); it!= Model.NATOM_ONETYPE.end(); it++){ 
     cout<<*it<<endl; 
     }
	cout<<endl;
	cout<<Model.ATOMS[0][0]<<endl;
	cout<<Model.ATOMS[0][1]<<endl;
	cout<<Model.ATOMS[0][2]<<endl;
	cout<<endl;
     cout << Model.XYZ[5][3] << endl;

     cout << Model.XYZ[7][11] << endl;




     return 0;
};


