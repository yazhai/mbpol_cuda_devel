#!/bin/bash

exefile="./bpnn_main.x"
inpfile="2b.xyz"

logfile="runtime.log"
outfile="runtime_statistics.out"
omp_thread_list=(1 2 4 8 16 24 32 36 48 72 96)         # Num of omp_threads list

itr_each_thread=5                      # Num of repeating runs for every omp_thread configuration
interested_labels="Gf_fwd_all NN_total "  # interested timer labels, separated by space


rm -f ${logfile} ${outfile}
## run bench with different threads

for ((j=0; j<${#omp_thread_list[@]} ; j++ ))
do
     omp_thread=${omp_thread_list[j]}
     export OMP_NUM_THREADS=${omp_thread}
     echo "OMP_NUM_THREADS =" $OMP_NUM_THREADS   >> $logfile
     echo "Running program with OMP THREADS" ${omp_thread_list[j]} 
     for ((i=1; i<=$itr_each_thread; i++))
     do
          echo "Iteration No. = "  $i >> $logfile
          ${exefile} ${inpfile} >> ${logfile} 2>/dev/null
     done
done


## Readout logfile and make statstics
awk -v utfile=${outfile} -v allfds="${interested_labels}" '
BEGIN{     
     nfds=split(allfds,fd_mtx," ");     
     omp_threads=0;
     first_time=1;        
     
     printf ( " Statistics of function accumulative runtime in [MuS]:\n           |") >> utfile ;
    
     for (i=1; i<=nfds; i++){
          pos = int( (30 + length(fd_mtx[i]))/2 );
          printf ( "%*s%*s", pos , fd_mtx[i], 30-pos, "|") >> utfile;
     }
     printf ( "\n OMP_THRDS |" ) >> utfile;
     for (i=1; i<=nfds; i++){
          printf ( "    MEAN       MIN       MAX |")>> utfile;
     }
     printf ( "\n" )>> utfile;
}

$1~/OMP_NUM_THREADS/ {      
     if (first_time!=1) {     
          printf ( "%6d     |", omp_threads)  >> utfile;
          for ( i=1; i<=nfds; i++ ) {
               var = fd_mtx[i];
               if (count[var] != 0){
                    ##ave = time_mtx[var]/count[var];
                    ave = time_mtx[var] ;
               } else {
                    ave = 0
               }               
               printf ( "%8.2g  %8.2g  %8.2g |", ave , minT_mtx[var], maxT_mtx[var]  )>> utfile;
          }
          printf ( "\n" ) >> utfile;                    
     } else {
          first_time=0;
     }   
     delete time_mtx;
     delete maxT_mtx;
     delete minT_mtx;    
     delete count; 
     omp_threads=$3;  
}

##$1~/Iteration/ {  iter[omp_threads] = $4;}

(!($1~/OMP_NUM_THREADS/)) && (!($1~/Iteration/)){
     count[$4]++;
     time_mtx[$4]+= $6;
     if (maxT_mtx[$4] < $6) { maxT_mtx[$4] = $6; }
     if ( (minT_mtx[$4] > $6) || (minT_mtx[$4] == 0) ) { minT_mtx[$4] = $6; }  
}

END{
     printf ( "%6d     |", omp_threads)  >> utfile;
     for ( i=1; i<=nfds; i++ ) {
          var = fd_mtx[i];
          if (count[var] != 0){
               ave = time_mtx[var]/count[var];
          } else {
               ave = 0
          }
          printf ( "%8.2g  %8.2g  %8.2g |", ave , minT_mtx[var], maxT_mtx[var]  )>> utfile;
     }
     printf ( "\n" ) >> utfile;
}' $logfile
     
     echo >> $outfile


cat ${outfile}



