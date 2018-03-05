

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



int main4(void){

     Gfunction_t<double> G;
     const char* xyzfile1 = "test.xyz";
     const char* xyzfile = "2b.xyz";
     const char* tmp = "";

     G.make_G_XYZ(xyzfile, tmp, tmp);


     return 0;
}
