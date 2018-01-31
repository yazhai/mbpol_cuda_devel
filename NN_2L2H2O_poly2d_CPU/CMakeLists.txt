cmake_minimum_required (VERSION 2.6)
project (Full_NN)

#select libraries to employ
option(_USE_GSL "use gsl_cblas library" 0)
option(_USE_MKL "use mkl_cblas library" 0)
option(_USE_OPENMP "use openmp library" 0)
option(_CMP_INTEL "use intel compiler" 0)
option(_CMP_GNU "use gnu c++ compiler" 1)

#set output folder
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})

#source code folder
set(SRC ${CMAKE_SOURCE_DIR}/src)

#set compiler
if(${_CMP_INTEL})
	SET(CMAKE_C_COMPILER icpc)
	SET(CMAKE_CXX_COMPILER icpc)
else(${_CMP_INTEL})
	SET(CMAKE_C_COMPILER g++)
	SET(CMAKE_CXX_COMPILER g++)
endif(${_CMP_INTEL})

#resource folder
set(RES_DIR ${CMAKE_SOURCE_DIR}/res/BPNN-2b)

#set all source files
set(SOURCES 	${SRC}/Gfunction.cpp 
		${SRC}/atomTypeID.cpp 
		${SRC}/network.cpp 
		${SRC}/readGparams.cpp
		${SRC}/timestamps.cpp 
		${SRC}/utility.cpp 
		${SRC}/atomTypeID_v2.h
		${SRC}/fullTester.cpp)
#set resource files
set(RESOURCES 	${RES_DIR}/Gfn_order.dat
	      	${RES_DIR}/Gfunc_params_2Bv14.dat 
		${RES_DIR}/2b_to_fit_rounded.xyz
		${RES_DIR}/max_per_feature)

file(COPY  ${RESOURCES} DESTINATION ${CMAKE_BINARY_DIR})

add_executable(Full_NN ${SOURCES})

#install(TARGETS Full_NN DESTINATION bin)

#compiler options
target_compile_options(Full_NN PUBLIC -std=c++11 -O3)

#set libraries
set(LIBS hdf5 hdf5_cpp)

if(${_USE_GSL})
	set(LIBS ${LIBS} -lgslcblas)
	add_definitions(-D_USE_GSL) 
endif(${_USE_GSL})

if(${_USE_OPENMP})
	set(LIBS ${LIBS} -fopenmp)
	add_definitions(-D_OPENMP)
endif(${_USE_OPENMP})

	
#libaries
target_link_libraries(Full_NN ${LIBS})

#clean outputs
ADD_CUSTOM_TARGET(cleanAll @echo cleaning all)
SET(CLEANEDAll		${CMAKE_BINARY_DIR}/gfn_o*
				${CMAKE_BINARY_DIR}/my_y_pred.txt
				${CMAKE_BINARY_DIR}/NN_final.out
				${CMAKE_BINARY_DIR}/Full_NN
				)

ADD_CUSTOM_COMMAND(
	DEPENDS clean
	COMMENT "clean all output files - only leaves input files"
	COMMAND rm -Rf ${CLEANEDAll}
	COMMAND make clean
	TARGET cleanAll
	)
