#include "readGparams_v2.h"


// nothing here but a tester
int main2(void){

     Gparams_t<double> GPARAM;

     const char* file = "O_ang";

     GPARAM.read_param_from_file(file);

     const char* file2 = "O_rad";
     GPARAM.read_param_from_file(file2);

     const char* file3 = "H_rad";
     GPARAM.read_param_from_file(file3);     


     return 0;
}