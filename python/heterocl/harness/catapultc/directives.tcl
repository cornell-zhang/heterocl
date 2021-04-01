    solution new -state initial
    solution options defaults
    solution file add /work/shared/users/phd/jl3952/installs/hcl-cal/samples/count_min/project/test.h -type CHEADER
    solution file add /work/shared/users/phd/jl3952/installs/hcl-cal/samples/count_min/project/kernel.cpp -type C++
    solution file add /work/shared/users/phd/jl3952/installs/hcl-cal/samples/count_min/project/testbench.cpp -type C++ -exclude true
    go analyze
    go compile 
    flow run /SCVerify/launch_make ./scverify/Verify_orig_cxx_osci.mk {} SIMTOOL=osci sim
    # go new
    solution library add nangate-45nm_beh -file     {$MGC_HOME/pkgs/siflibs/nangate/nangate-45nm_beh.lib} --     -rtlsyntool DesignCompiler
    solution library add ccs_sample_mem -file     {$MGC_HOME/pkgs/siflibs/ccs_sample_mem.lib}
    solution library add amba
    solution library add ML_amba
    go libraries
    directive set -CLOCKS {clk {-CLOCK_PERIOD 2.0 -CLOCK_EDGE rising     -CLOCK_UNCERTAINTY 0.0 -CLOCK_HIGH_TIME 1.0 -RESET_SYNC_NAME     rst -RESET_ASYNC_NAME arst_n -RESET_KIND sync -RESET_SYNC_ACTIVE     high -RESET_ASYNC_ACTIVE low -ENABLE_ACTIVE high}}
    go assembly
    go architect
    go allocate
    go extract
    # flow run /SCVerify/launch_make ./scverify/Verify_rtl_v_msim.mk {} SIMTOOL=msim simgui
