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

#ifndef TB_H
#define TB_H


#include <cynw_p2p.h>	// The P2P port header file

SC_MODULE( tb )
{
	sc_in< bool >	clk;		// Declare clock and reset ports
	sc_out< bool >	rst;		// The source thread drives rst_out.
	sc_in<bool>     finish;

	// The cynw_p2p ::base_in and ::base_out port classes are used in
	// the testbench. These are simpler versions of the ::in and ::out
	// classes used in the DUT.
	//cynw_p2p < input_t >::base_out dout_1; // data going in to the design
	//cynw_p2p < input_t >::base_out dout_2; // data going in to the design
	//cynw_p2p < output_t >::base_in din;  // data going out of the design
	cynw_p2p < sc_uint<32> >::base_out instr_phy_addr;
	cynw_p2p < sc_uint<32> >::base_out instr_count;

	// memories
	sc_uint<8>*	  DRAM;

	// DRAM length
	int len;
	// read buffer
	unsigned char* buff; 

	SC_HAS_PROCESS(tb);
	tb( sc_module_name name , sc_uint<8>	_DRAM[268435456] )
	: clk( "clk" )
	, rst( "rst" )
	, instr_phy_addr("instr_phy_addr")
	, finish("finish")
	, instr_count("instr_count")
  	, DRAM(_DRAM)	
	{
		// Declare the source and sink threads. Since they 
		// drive the reset signal, no reset_signal_is() is needed
		SC_CTHREAD( source, clk.pos() );
		SC_CTHREAD( sink, clk.pos() );
	}
  
	void source();
	void sink();

};

#endif // TB_H
