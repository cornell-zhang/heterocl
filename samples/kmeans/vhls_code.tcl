set hls_prj out.prj
open_project ${hls_prj} -reset

set_top default_function
add_files vhls_code.cpp

open_solution "solution1"
set_part xcvu9p-fsgd2104-2-i

csynth_design
