#!/bin/bash

reffile_list=("2b_to_fit_xyz_energy.ref" )
rstfile_list=("E_nogrd.rst" )

err_threshold=1e-6

for ((j=0; j<${#reffile_list[@]} ; j++ ))
do
     reffile=${reffile_list[j]}
     rstfile=${rstfile_list[j]}

echo Benchmark $rstfile against $reffile ...... 

## Readout logfile and make statstics
awk -v err_thrd=${err_threshold} '
function abs(v) {return v < 0 ? -v : v} ;
BEGIN{     
     rec_num = 0 ;
     rst_idx = 0 ; 
     max_err = 0 ;
     tot_err = 0 ;
}

FNR==NR {
     for ( i=1; i<=NF; i++) {
          rec_num++;
          ref_val[rec_num]=$i;
     }
}

FNR!=NR{
     for ( i=1; i<=NF; i++) {
          rst_idx++;
          err =  abs( $i - ref_val[rst_idx] );
          tot_err+=err^2;
          if(err > max_err) {max_err = err};
     }
}

END{
    mean_sqrt_err = sqrt(tot_err/rec_num);
    if (max_err < err_thrd) {
          result = "passed";
    } else{
          result = "failed";
    }
    printf ( "  Max err: %10.8e  ; Mean sqrt err: %10.8e  ; Test %s!!!\n" , max_err, mean_sqrt_err,result );
}' $reffile $rstfile

echo 

done




