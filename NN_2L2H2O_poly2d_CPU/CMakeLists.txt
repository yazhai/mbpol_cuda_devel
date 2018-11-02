cmake_minimum_required (VERSION 2.6)

#select libraries to employ
option(GSL "use gsl_cblas library" 0)
option(MKL "use mkl_cblas library, only compatible with icpc" 0)
option(OPENMP "use openmp library" 0)
option(INTEL "use intel compiler" 0)
option(GNU "use gnu c++ compiler" 1)

option(XHOST "use xHost optimization for icpc" 0)
option(O "optimization level" -1)

option(g "debugging" 0)

#set compiler
if(INTEL)
	SET(CMAKE_C_COMPILER icc)
	SET(CMAKE_CXX_COMPILER icpc)
else()
	SET(CMAKE_C_COMPILER gcc)
	SET(CMAKE_CXX_COMPILER g++)
endif()

#start project
project (bpnn_main.x)


#set output folder
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})

message(${CMAKE_SOURCE_DIR})
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


install(TARGETS bpnn_main.x DESTINATION ${CMAKE_BINARY_DIR})


#compiler options
target_compile_options(bpnn_main.x PUBLIC -std=c++11)

#set libraries
set(LIBS hdf5 hdf5_cpp)

if(MKL)
     if(INTEL)
	     set(LIBS ${LIBS} -mkl)
	     #add_definitions(-D_MKL) 
	else()
	     message("Set using GSL ...... \n Please set MKL with icpc compiler")
     	set(LIBS ${LIBS} -lgslcblas)
	     #add_definitions(-D_GSL)
	endif()
elseif(GSL)
	set(LIBS ${LIBS} -lgslcblas)
	#add_definitions(-D_USE_GSL) 
endif()


if(OPENMP)
     if(INTEL)
	     target_compile_options(bpnn_main.x PUBLIC -qopenmp )
	     set(LIBS ${LIBS} -qopenmp)
	     #add_definitions(-qopenmp)
	else()
          target_compile_options(bpnn_main.x PUBLIC -fopenmp )
          set(LIBS ${LIBS} -fopenmp)
          #add_definitions(-fopenmp)
	endif()
endif()


if(NOT ${O} EQUAL -1)
     if(${O} EQUAL 0)
	     target_compile_options(bpnn_main.x PUBLIC -O0 )
	     set(LIBS ${LIBS} -O0)
	     #add_definitions(-O0)     
	elseif(${O} EQUAL 1)
	     target_compile_options(bpnn_main.x PUBLIC -O1 )
	     set(LIBS ${LIBS} -O1)
	     #add_definitions(-O1)      
	elseif(${O} EQUAL 2)
	     target_compile_options(bpnn_main.x PUBLIC -O2 )
	     set(LIBS ${LIBS} -O2)
	     #add_definitions(-O2)   
	elseif(${O} EQUAL 3)
	     target_compile_options(bpnn_main.x PUBLIC -O3 )
	     set(LIBS ${LIBS} -O3)
	     #add_definitions(-O3)   	     	     
     endif()
endif()

if (XHOST)
     if (INTEL)
          target_compile_options(bpnn_main.x PUBLIC -xHost )
          set(LIBS ${LIBS} -xHost)
          #add_definitions(-xHost)
     endif()
endif()


if (g)
     target_compile_options(bpnn_main.x PUBLIC -g)
endif()



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
	COMMAND rm -rf ${CLEANEDAll}
	COMMAND make clean
	TARGET cleanAll
	)
