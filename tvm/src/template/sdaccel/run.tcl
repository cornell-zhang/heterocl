set hls_prj digitrec.prj
open_project ${hls_prj} -reset
set_top default_function
add_files -tb main.cpp
add_files -tb data

open_solution "solution1"
set_part {xc7z020clg484-1}
create_clock -period 10

csim_design -O
csynth_design
#cosim_design
exit
