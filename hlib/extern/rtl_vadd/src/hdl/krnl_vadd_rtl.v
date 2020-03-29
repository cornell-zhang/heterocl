// /*******************************************************************************
// Copyright (c) 2018, Xilinx, Inc.
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
// 
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// 
// 
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// 
// 
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
// 
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,THE IMPLIED 
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
// IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, 
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY 
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
// EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// *******************************************************************************/

///////////////////////////////////////////////////////////////////////////////
// Description: This is a wrapper of module "krnl_vadd_rtl_int"
///////////////////////////////////////////////////////////////////////////////

// default_nettype of none prevents implicit wire declaration.
`default_nettype none
`timescale 1 ns / 1 ps 

module krnl_vadd_rtl #( 
  parameter integer  C_S_AXI_CONTROL_DATA_WIDTH = 32,
  parameter integer  C_S_AXI_CONTROL_ADDR_WIDTH = 6,
  parameter integer  C_M_AXI_GMEM_ID_WIDTH = 1,
  parameter integer  C_M_AXI_GMEM_ADDR_WIDTH = 64,
  parameter integer  C_M_AXI_GMEM_DATA_WIDTH = 32
)
(
  // System signals
  input  wire  ap_clk,
  input  wire  ap_rst_n,
  // AXI4 master interface 
  output wire                                 m_axi_gmem_AWVALID,
  input  wire                                 m_axi_gmem_AWREADY,
  output wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0]   m_axi_gmem_AWADDR,
  output wire [C_M_AXI_GMEM_ID_WIDTH - 1:0]   m_axi_gmem_AWID,
  output wire [7:0]                           m_axi_gmem_AWLEN,
  output wire [2:0]                           m_axi_gmem_AWSIZE,
  // Tie-off AXI4 transaction options that are not being used.
  output wire [1:0]                           m_axi_gmem_AWBURST,
  output wire [1:0]                           m_axi_gmem_AWLOCK,
  output wire [3:0]                           m_axi_gmem_AWCACHE,
  output wire [2:0]                           m_axi_gmem_AWPROT,
  output wire [3:0]                           m_axi_gmem_AWQOS,
  output wire [3:0]                           m_axi_gmem_AWREGION,
  output wire                                 m_axi_gmem_WVALID,
  input  wire                                 m_axi_gmem_WREADY,
  output wire [C_M_AXI_GMEM_DATA_WIDTH-1:0]   m_axi_gmem_WDATA,
  output wire [C_M_AXI_GMEM_DATA_WIDTH/8-1:0] m_axi_gmem_WSTRB,
  output wire                                 m_axi_gmem_WLAST,
  output wire                                 m_axi_gmem_ARVALID,
  input  wire                                 m_axi_gmem_ARREADY,
  output wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0]   m_axi_gmem_ARADDR,
  output wire [C_M_AXI_GMEM_ID_WIDTH-1:0]     m_axi_gmem_ARID,
  output wire [7:0]                           m_axi_gmem_ARLEN,
  output wire [2:0]                           m_axi_gmem_ARSIZE,
  output wire [1:0]                           m_axi_gmem_ARBURST,
  output wire [1:0]                           m_axi_gmem_ARLOCK,
  output wire [3:0]                           m_axi_gmem_ARCACHE,
  output wire [2:0]                           m_axi_gmem_ARPROT,
  output wire [3:0]                           m_axi_gmem_ARQOS,
  output wire [3:0]                           m_axi_gmem_ARREGION,
  input  wire                                 m_axi_gmem_RVALID,
  output wire                                 m_axi_gmem_RREADY,
  input  wire [C_M_AXI_GMEM_DATA_WIDTH - 1:0] m_axi_gmem_RDATA,
  input  wire                                 m_axi_gmem_RLAST,
  input  wire [C_M_AXI_GMEM_ID_WIDTH - 1:0]   m_axi_gmem_RID,
  input  wire [1:0]                           m_axi_gmem_RRESP,
  input  wire                                 m_axi_gmem_BVALID,
  output wire                                 m_axi_gmem_BREADY,
  input  wire [1:0]                           m_axi_gmem_BRESP,
  input  wire [C_M_AXI_GMEM_ID_WIDTH - 1:0]   m_axi_gmem_BID,

  // AXI4-Lite slave interface
  input  wire                                    s_axi_control_AWVALID,
  output wire                                    s_axi_control_AWREADY,
  input  wire [C_S_AXI_CONTROL_ADDR_WIDTH-1:0]   s_axi_control_AWADDR,
  input  wire                                    s_axi_control_WVALID,
  output wire                                    s_axi_control_WREADY,
  input  wire [C_S_AXI_CONTROL_DATA_WIDTH-1:0]   s_axi_control_WDATA,
  input  wire [C_S_AXI_CONTROL_DATA_WIDTH/8-1:0] s_axi_control_WSTRB,
  input  wire                                    s_axi_control_ARVALID,
  output wire                                    s_axi_control_ARREADY,
  input  wire [C_S_AXI_CONTROL_ADDR_WIDTH-1:0]   s_axi_control_ARADDR,
  output wire                                    s_axi_control_RVALID,
  input  wire                                    s_axi_control_RREADY,
  output wire [C_S_AXI_CONTROL_DATA_WIDTH-1:0]   s_axi_control_RDATA,
  output wire [1:0]                              s_axi_control_RRESP,
  output wire                                    s_axi_control_BVALID,
  input  wire                                    s_axi_control_BREADY,
  output wire [1:0]                              s_axi_control_BRESP 
);

krnl_vadd_rtl_int #(
  .C_S_AXI_CONTROL_DATA_WIDTH  ( C_S_AXI_CONTROL_DATA_WIDTH ),
  .C_S_AXI_CONTROL_ADDR_WIDTH  ( C_S_AXI_CONTROL_ADDR_WIDTH ),
  .C_M_AXI_GMEM_ID_WIDTH       ( C_M_AXI_GMEM_ID_WIDTH ),
  .C_M_AXI_GMEM_ADDR_WIDTH     ( C_M_AXI_GMEM_ADDR_WIDTH ),
  .C_M_AXI_GMEM_DATA_WIDTH     ( C_M_AXI_GMEM_DATA_WIDTH )
)
inst_krnl_vadd_rtl_int (
  .ap_clk                 ( ap_clk ),
  .ap_rst_n               ( ap_rst_n ),
  .m_axi_gmem_AWVALID     ( m_axi_gmem_AWVALID ),
  .m_axi_gmem_AWREADY     ( m_axi_gmem_AWREADY ),
  .m_axi_gmem_AWADDR      ( m_axi_gmem_AWADDR ),
  .m_axi_gmem_AWID        ( m_axi_gmem_AWID ),
  .m_axi_gmem_AWLEN       ( m_axi_gmem_AWLEN ),
  .m_axi_gmem_AWSIZE      ( m_axi_gmem_AWSIZE ),
  .m_axi_gmem_AWBURST     ( m_axi_gmem_AWBURST ),
  .m_axi_gmem_AWLOCK      ( m_axi_gmem_AWLOCK ),
  .m_axi_gmem_AWCACHE     ( m_axi_gmem_AWCACHE ),
  .m_axi_gmem_AWPROT      ( m_axi_gmem_AWPROT ),
  .m_axi_gmem_AWQOS       ( m_axi_gmem_AWQOS ),
  .m_axi_gmem_AWREGION    ( m_axi_gmem_AWREGION ),
  .m_axi_gmem_WVALID      ( m_axi_gmem_WVALID ),
  .m_axi_gmem_WREADY      ( m_axi_gmem_WREADY ),
  .m_axi_gmem_WDATA       ( m_axi_gmem_WDATA ),
  .m_axi_gmem_WSTRB       ( m_axi_gmem_WSTRB ),
  .m_axi_gmem_WLAST       ( m_axi_gmem_WLAST ),
  .m_axi_gmem_ARVALID     ( m_axi_gmem_ARVALID ),
  .m_axi_gmem_ARREADY     ( m_axi_gmem_ARREADY ),
  .m_axi_gmem_ARADDR      ( m_axi_gmem_ARADDR ),
  .m_axi_gmem_ARID        ( m_axi_gmem_ARID ),
  .m_axi_gmem_ARLEN       ( m_axi_gmem_ARLEN ),
  .m_axi_gmem_ARSIZE      ( m_axi_gmem_ARSIZE ),
  .m_axi_gmem_ARBURST     ( m_axi_gmem_ARBURST ),
  .m_axi_gmem_ARLOCK      ( m_axi_gmem_ARLOCK ),
  .m_axi_gmem_ARCACHE     ( m_axi_gmem_ARCACHE ),
  .m_axi_gmem_ARPROT      ( m_axi_gmem_ARPROT ),
  .m_axi_gmem_ARQOS       ( m_axi_gmem_ARQOS ),
  .m_axi_gmem_ARREGION    ( m_axi_gmem_ARREGION ),
  .m_axi_gmem_RVALID      ( m_axi_gmem_RVALID ),
  .m_axi_gmem_RREADY      ( m_axi_gmem_RREADY ),
  .m_axi_gmem_RDATA       ( m_axi_gmem_RDATA ),
  .m_axi_gmem_RLAST       ( m_axi_gmem_RLAST ),
  .m_axi_gmem_RID         ( m_axi_gmem_RID ),
  .m_axi_gmem_RRESP       ( m_axi_gmem_RRESP ),
  .m_axi_gmem_BVALID      ( m_axi_gmem_BVALID ),
  .m_axi_gmem_BREADY      ( m_axi_gmem_BREADY ),
  .m_axi_gmem_BRESP       ( m_axi_gmem_BRESP ),
  .m_axi_gmem_BID         ( m_axi_gmem_BID ),
  .s_axi_control_AWVALID  ( s_axi_control_AWVALID ),
  .s_axi_control_AWREADY  ( s_axi_control_AWREADY ),
  .s_axi_control_AWADDR   ( s_axi_control_AWADDR ),
  .s_axi_control_WVALID   ( s_axi_control_WVALID ),
  .s_axi_control_WREADY   ( s_axi_control_WREADY ),
  .s_axi_control_WDATA    ( s_axi_control_WDATA ),
  .s_axi_control_WSTRB    ( s_axi_control_WSTRB ),
  .s_axi_control_ARVALID  ( s_axi_control_ARVALID ),
  .s_axi_control_ARREADY  ( s_axi_control_ARREADY ),
  .s_axi_control_ARADDR   ( s_axi_control_ARADDR ),
  .s_axi_control_RVALID   ( s_axi_control_RVALID ),
  .s_axi_control_RREADY   ( s_axi_control_RREADY ),
  .s_axi_control_RDATA    ( s_axi_control_RDATA ),
  .s_axi_control_RRESP    ( s_axi_control_RRESP ),
  .s_axi_control_BVALID   ( s_axi_control_BVALID ),
  .s_axi_control_BREADY   ( s_axi_control_BREADY ),
  .s_axi_control_BRESP    ( s_axi_control_BRESP )
);
endmodule : krnl_vadd_rtl

`default_nettype wire
