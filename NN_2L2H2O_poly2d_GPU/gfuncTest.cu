#include <cstdlib>
#include <vector>
#include <string>
#include <map>
#include <limits>

#include "readGparams_v2.h"
#include "atomTypeID_v2.h"
#include "timestamps.h"
#include "utility.h"
#include "utility_cu.cuh"

using namespace std;



__device__ double getDist(double * xyz0, double * xyz1){
          double a = xyz0[0] - xyz1[0];
		double b = xyz0[1] - xyz1[1];
		double c = xyz0[2] - xyz1[2];
          double dist = sqrt(a*a + b*b + c*c);
          return dist;
}

//xyz0 and xyz1 are pointers to device memory-- where the xyz coordinates of atom 0 and 1 begin
__global__ void getRadial(double * xyz0, double * xyz1){}


atom_Type_ID_t<double> model;



void load_xyz(const char* file){
     model.load_xyz(file);
};


int main(){

     atom_Type_ID_t<double> model;

     cout <<" Starting Test"<<endl;
     load_xyz("2b_test.xyz");

     //want: atom1: x y z x y z x y z x y z
     //      atom2: x y z x y z x y z x y z...
     double ** xyz = nullptr;
     init_mtx_in_mem(xyz, 6, 6);
     for(int i = 0; i < 6; i ++){
          for(int j = 0; j < 2; j++){
               xyz[i][j*3] = i*(j+1);
               xyz[i][j*3+1] = i*(j+1);
               xyz[i][j*3+2] = i*(j+1);
          }
     }

     double * xyz_d = nullptr;
     size_t pitch = 0;
     memcpy_mtx_h2d(xyz_d, pitch , xyz, 6, 6);

     cout<<pitch<<endl;

     printDeviceMatrix(xyz_d, pitch, 6 , 6);



     return 0;
}