

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



int main4(int argc, char *argv[]){

     Gfunction_t<double> G;
     const char* xyzfile1 = "test.xyz";
     const char* xyzfile = "2b.xyz";
     const char* tmp = "";

     G.load_xyz("test1.xyz");
     G.load_paramfile("Gfunc_params_2Bv14_tuned.dat");
     G.load_seq("Gfn_order.dat");

     G.make_G();

     G.make_G();

     std::vector<double**> dfdG;
     double** grdO = nullptr;

     std::ofstream ut;
     // ut.open("grd_set.rst");

     for(int i = 0; i < 2; i++){
          double ** grd = nullptr;
          init_mtx_in_mem(grd, 1, 82);
          // for(int j=0; j < 82; j++) {
          //      grdO[0][i] = G.G[i][0][j]*scale;
          //      ut << std::scientific << std::setprecision(18) << grdO[0][i] << " ";
          // }
          // ut << std::endl;
          dfdG.push_back(grd);
     }

     for(int i = 2; i < 6; i++){
          double ** grd = nullptr;
          init_mtx_in_mem(grd, 1, 84);
          // for(int j=0; j < 84; j++) {
          //      grdO[0][i] = G.G[i][0][j]*scale;
          //      ut << std::scientific << std::setprecision(18) << grdO[0][i] << " ";
          // }
          // ut << std::endl;
          dfdG.push_back(grd);
     }
     
     int checka = 0;
     int checkx = 0;

     int checkg = 2;
     int checkp = 75;

     checka = std::stoi(argv[1]);
     checkx = std::stoi(argv[2]);
     checkg = std::stoi(argv[3]);
     checkp = std::stoi(argv[4]);
     double scale = 0.001;


     dfdG[checkg][0][checkp]=1.0;

     // ut.close();

     G.make_grd(dfdG);

     std::cout << "Gradient from prog is " << std::scientific << std::setprecision(18) <<G.dfdxyz[0][checka*3 + checkx]  << std::endl;



     double p_old = G.G[checkg][0][checkp] ;

     double dx = G.model.XYZ[0][checka*3 + checkx] * scale ;
     G.model.XYZ[0][checka*3 + checkx] += dx ;
     G.make_G();

     double p_new = G.G[checkg][0][checkp] ;

     std::cout << "Gradient from test is " << std::scientific << std::setprecision(18) << (p_new - p_old) / dx  << std::endl;     





     // ut.open("grd_xyz.rst");
     // for(int i = 0; i< 6; i++){
     //      for(int j = 0; j<3; j++) {
     //           ut << std::scientific << std::setprecision(18) << G.dfdxyz[0][i*3+j] << " ";
     //           G.model.XYZ[0][i*3+j] += G.dfdxyz[0][i*3+j] ;
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
     // for(int n = 0; n< G.NType ; n++){
     //      for(int ii =0 ; ii< G.TypeNAtom[n]; ii++ ){
     //           int np = G.G_param_max_size[n];
     //           for (int c = 0; c<G.NCluster; c++){
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
