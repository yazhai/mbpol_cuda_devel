cmake_minimum_required (VERSION 2.6)
project (bpnn_main.x)

#select libraries to employ
option(_USE_GSL "use gsl_cblas library" 0)
option(_USE_MKL "use mkl_cblas library" 0)
option(_USE_OPENMP "use openmp library" y)
option(_CMP_INTEL "use intel compiler" 0)
option(_CMP_GNU "use gnu c++ compiler" 1)

#set output folder
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})

message(${CMAKE_SOURCE_DIR})


#set compiler
if(${_CMP_INTEL})
	SET(CMAKE_C_COMPILER icpc)
	SET(CMAKE_CXX_COMPILER icpc)
else(${_CMP_INTEL})
	SET(CMAKE_C_COMPILER g++)
	SET(CMAKE_CXX_COMPILER g++)
endif(${_CMP_INTEL})

#resource folder
set(RES_DIR ${CMAKE_SOURCE_DIR}/res)

#resource files
file(GLOB RES "${RES_DIR}/*.*" "${RES_DIR}/BPNN-2b/*" "${RES_DIR}/BPNN-3b/*")

#source code folder
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)

#set all source files
file(GLOB SRC "${SRC_DIR}/*")

file(COPY  ${RES} DESTINATION ${CMAKE_BINARY_DIR})

add_executable(bpnn_main.x ${SRC})

include_directories(${CMAKE_SOURCE_DIR}/inc)


#install(TARGETS bpnn_main.x DESTINATION bin)

#compiler options
target_compile_options(bpnn_main.x PUBLIC -g -std=c++11 -O0)

#set libraries
set(LIBS hdf5 hdf5_cpp)

if(${_USE_GSL})
	set(LIBS ${LIBS} -lgslcblas)
	add_definitions(-D_USE_GSL) 
endif(${_USE_GSL})

if(${_USE_OPENMP})
	set(LIBS ${LIBS} -fopenmp)
	add_definitions(-D_OPENMP -fopenmp)
endif(${_USE_OPENMP})

set(LIBS ${LIBS} -fopenmp)
add_definitions(-D_OPENMP -fopenmp)

	
#libaries
target_link_libraries(bpnn_main.x ${LIBS})

#clean outputs
ADD_CUSTOM_TARGET(cleanAll @echo cleaning all)
SET(CLEANEDAll		${CMAKE_BINARY_DIR}/gfn_o*
				${CMAKE_BINARY_DIR}/my_y_pred.txt
				${CMAKE_BINARY_DIR}/NN_final.out
				${CMAKE_BINARY_DIR}/bpnn_main.x
				)

ADD_CUSTOM_COMMAND(
	DEPENDS clean
	COMMENT "clean all output files - only leaves input files"
	COMMAND rm -Rf ${CLEANEDAll}
	COMMAND make clean
	TARGET cleanAll
	)
