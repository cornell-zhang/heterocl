///////////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2017 Cadence Design Systems, Inc. All rights reserved worldwide.
//
// The code contained herein is the proprietary and confidential information
// of Cadence or its licensors, and is supplied subject to a previously
// executed license and maintenance agreement between Cadence and customer.
// This code is intended for use with Cadence high-level synthesis tools and
// may not be used with other high-level synthesis tools. Permission is only
// granted to distribute the code as indicated. Cadence grants permission for
// customer to distribute a copy of this code to any partner to aid in designing
// or verifying the customer's intellectual property, as long as such
// distribution includes a restriction of no additional distributions from the
// partner, unless the partner receives permission directly from Cadence.
//
// ALL CODE FURNISHED BY CADENCE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY
// KIND, AND CADENCE SPECIFICALLY DISCLAIMS ANY WARRANTY OF NONINFRINGEMENT,
// FITNESS FOR A PARTICULAR PURPOSE OR MERCHANTABILITY. CADENCE SHALL NOT BE
// LIABLE FOR ANY COSTS OF PROCUREMENT OF SUBSTITUTES, LOSS OF PROFITS,
// INTERRUPTION OF BUSINESS, OR FOR ANY OTHER SPECIAL, CONSEQUENTIAL OR
// INCIDENTAL DAMAGES, HOWEVER CAUSED, WHETHER FOR BREACH OF WARRANTY,
// CONTRACT, TORT, NEGLIGENCE, STRICT LIABILITY OR OTHERWISE.
//
////////////////////////////////////////////////////////////////////////////////

#include <systemc.h>            	// SystemC definitions
#include "system.h"             	// Top-level System module header file

static System * m_system = NULL;	// The pointer that holds the top-level System module instance.

void esc_elaborate()			// This function is required by Stratus to support SystemC-Verilog
{					// cosimulation. It instances the top-level module.
	m_system = new System( "system" );
}

void esc_cleanup()			// This function is called at the end of simulation by the
{					// Stratus co-simulation hub. It should delete the top-level
	delete m_system;		// module instance.
}

int sc_main( int argc, char ** argv )	// This function is called by the SystemC kernel for pure SystemC simulations
{
	esc_initialize( argc, argv );	// esc_initialize() passes in the cmd-line args. This initializes the Stratus simulation 
					// environment (such as opening report files for later logging and analysis).

	esc_elaborate();		// esc_elaborate() (defined above) creates the top-level module instance. In a SystemC-Verilog 
					// co-simulation, this is called during cosim initialization rather than from sc_main.

    sc_start();			// Starts the simulation. Returns when a module calls esc_stop(), which finishes the simulation.
    					// esc_cleanup() (defined above) is automatically called before sc_start() returns.

	return 0;			// Returns the status of the simulation. Required by most C compilers.
}
