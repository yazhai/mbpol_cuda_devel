#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <memory>
#include <cstdlib>


#include "readGparams_cu.cuh"


using namespace std;


template<>
double Gparams_t<double>::return_a_number(string _string){
     return  stod ( _string );     
}

template<>
float Gparams_t<float>::return_a_number(string _string){
     return  stof ( _string );     
}


