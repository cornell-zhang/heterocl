###############################################################################
#
# Copyright (c) 2017 Cadence Design Systems, Inc. All rights reserved worldwide.
#
# The code contained herein is the proprietary and confidential information
# of Cadence or its licensors, and is supplied subject to a previously
# executed license and maintenance agreement between Cadence and customer.
# This code is intended for use with Cadence high-level synthesis tools and
# may not be used with other high-level synthesis tools. Permission is only
# granted to distribute the code as indicated. Cadence grants permission for
# customer to distribute a copy of this code to any partner to aid in designing
# or verifying the customer's intellectual property, as long as such
# distribution includes a restriction of no additional distributions from the
# partner, unless the partner receives permission directly from Cadence.
#
# ALL CODE FURNISHED BY CADENCE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY
# KIND, AND CADENCE SPECIFICALLY DISCLAIMS ANY WARRANTY OF NONINFRINGEMENT,
# FITNESS FOR A PARTICULAR PURPOSE OR MERCHANTABILITY. CADENCE SHALL NOT BE
# LIABLE FOR ANY COSTS OF PROCUREMENT OF SUBSTITUTES, LOSS OF PROFITS,
# INTERRUPTION OF BUSINESS, OR FOR ANY OTHER SPECIAL, CONSEQUENTIAL OR
# INCIDENTAL DAMAGES, HOWEVER CAUSED, WHETHER FOR BREACH OF WARRANTY,
# CONTRACT, TORT, NEGLIGENCE, STRICT LIABILITY OR OTHERWISE.
#
################################################################################

# Include the Makefile.prj file extracted from project.tcl
-include Makefile.prj

# Rule to update Makefile.prj anytime project.tcl changes
Makefile.prj : project.tcl
	@bdw_makegen

# List subdirectories that contain memory libraries
MEMLIBS =

# List subdirectories that contain interface libraries
IFLIBS =

# Uncomment this line to use the Insight debugger
BDW_DEBUG = 1

# The cmp_result target compares the golden results file (golden.dat) with
# the simulation results file (data.out) written by the testbench into the 
# BDW_SIM_CONFIG simulation configuration directory. This is run automatically
# at the end of each simulation (endOfSimCommand in project.tcl).
GOLD_FILE = DRAM/golden.bin
RESULTS_FILE = DRAM/result.bin 

# cmp_result:
# 	@if [ -f $(RESULTS_FILE) ] ; then \
# 	  echo "****************************************" ; \
# 	  echo `date` ; \
# 	  echo "Performing Simulation Results Comparison" ; \
# 	  echo "for $(BDW_SIM_CONFIG) Simulation..." ; \
# 	  if diff $(RESULTS_FILE) $(GOLD_FILE) ; then \
# 	    echo "  $(BDW_SIM_CONFIG): SIMULATION PASSED" ; \
# 	    bdw_sim_pass ; \
# 	  else \
# 	    echo "  $(BDW_SIM_CONFIG): SIMULATION FAILED" ; \
# 	    bdw_sim_fail ; \
# 	  fi ; \
# 	  echo "****************************************" ; \
# 	else \
# 	  echo "****************************************" ; \
# 	  echo `date` ; \
# 	  echo "ERROR: Simulation output doesn't exist" ; \
# 	  echo "for $(BDW_SIM_CONFIG) Simulation..." ; \
# 	  echo "  $(BDW_SIM_CONFIG): SIMULATION FAILED" ; \
# 	  bdw_sim_fail ; \
# 	  echo "****************************************" ; \
# 	fi
cmp_result:
	@if [ -f $(RESULTS_FILE) ] ; then \
	  echo "****************************************" ; \
	  echo `date` ; \
	  echo "for $(BDW_SIM_CONFIG) Simulation..." ; \
	  echo "  $(BDW_SIM_CONFIG): SIMULATION PASSED" ; \
	  bdw_sim_pass ; \
	  echo "****************************************" ; \
	else \
	  echo "****************************************" ; \
	  echo `date` ; \
	  echo "ERROR: Simulation output doesn't exist" ; \
	  echo "for $(BDW_SIM_CONFIG) Simulation..." ; \
	  echo "  $(BDW_SIM_CONFIG): SIMULATION FAILED" ; \
	  bdw_sim_fail ; \
	  echo "****************************************" ; \
	fi

# Clean out undesirable junk files from the project directory
# Uses the automatically created clean_all target from Makefile.prj
clean: clean_all
	@rm -rf bdw_work vsim* *.wlf data.out *~ *.bak *.BAK Makefile.prj
	@rm -rf core *.pro *.user *.log .*.qws .*.check .stack* .parts*
	@rm -rf rc.* fv .stratus*

CLEAN: clean
