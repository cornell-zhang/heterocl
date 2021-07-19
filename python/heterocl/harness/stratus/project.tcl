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

#
# Libraries
#
set LIB_PATH    "[get_install_path]/share/stratus/techlibs/GPDK045/gsclib045_svt_v4.4/gsclib045/timing"
set LIB_NAME	"slow_vdd1v2_basicCells.lib"
use_tech_lib	"$LIB_PATH/$LIB_NAME"

use_hls_lib "./memlib"


#
# C++ compiler options
#
set CLOCK_PERIOD "20.0"
set_attr cc_options			" -g -DCLOCK_PERIOD=$CLOCK_PERIOD"

#
# stratus_hls options
#
set_attr clock_period			$CLOCK_PERIOD	;# Clock period is required
#set_attr cycle_slack			1.75			;# Cycle slack adds margin to help logic synthesis close timing
set_attr balance_expr			off				;# Control expression balancing: off, width, delay  default=off
set_attr default_input_delay	0.1				;# Prevents registering of inputs
set_attr dpopt_auto				off				;# Set automatic DPOPT part creation: off, array, op, expr
set_attr dpopt_with_enable		off				;# Control if enable pins are on DPOPT parts (default=off)
set_attr flatten_arrays			none			;# Control array flattening: none, all_const, lhs_const, all default=none
set_attr global_state_encoding	binary			;# Control FSM state encoding: binary, one_hot  default=binary
#set_attr ignore_cells			"XYZ* ABC*"		;# Define list of cells to ignore
set_attr inline_partial_constants	on			;# Remove constant portions of variables
set_attr lsb_trimming			off				;# Remove unused LSBs: default=off
set_attr message_detail			1				;# Detail level in log file: 0, 1, or 2
set_attr method_processing		synthesize		;# Control SC_METHOD synthesis: default=translate, recommended=synthesize
set_attr path_delay_limit		120				;# Prevents long paths through resource sharing muxes (in % of clock period)
set_attr sched_asap				off				;# Create shortest possible schedule: default=off, recommended=off
set_attr unroll_loops			off				;# Control loop unrolling: default=off
set_attr wireload				none			;# Control wireload model used: default=none
# set_attr fuse_loops             off             ;

#
# Simulation Options
#
use_verilog_simulator   xcelium				;# 'mti', 'incisive', 'vcs', 'vcsi'
# enable_waveform_logging	-vcd					;# Generate vcd or fsdb waveform file with each simulation
set_attr end_of_sim_command "make cmp_result"	;# Make rule to run at end of each simulation

#
# System Module Configurations
#
define_system_module	main	main.cc		;# SystemC module for simulation only
define_system_module	System	system.cc	;# SystemC module for simulation only
define_system_module	tb	tb.cc			;# SystemC module for simulation only

#
# Synthesis Module Configurations
#
define_hls_module	dut	dut.cc		;# SystemC module to be synthesized
define_hls_config	dut	BASIC		;# A synthesis configuration of the hls_module
# set_logic_synthesis_options {BDW_LS_NOTOUCH_ALL};

## External array access
define_external_array_access -to System  -from dut

#
# Simulation Configurations
#
define_sim_config B			{dut BEH}			;# A behavioral PIN-level configuration
define_sim_config BASIC_V	{dut RTL_V BASIC}	;# RTL Verilog cosimulation

######################################################################
# Genus Logic Synthesis Configuration
######################################################################
define_logic_synthesis_config L {dut -all} \
    -options \
        {BDW_LS_NOGATES 1} \
        {BDW_LS_DO_DISSOLVE 1} \
        {BDW_LS_NOTOUCH_ALL 1} \
    -command \
        bdw_runrc

