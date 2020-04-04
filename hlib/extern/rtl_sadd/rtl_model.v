// 67d7842dbbe25473c3c32b313c0da8047785f30d78e8a024de1b57352245f316831
// This module describes SIMD Inference 
// 4 adders packed into single DSP block
`timescale 100ps/100ps

(* use_dsp = "simd" *)
(* dont_touch = "1" *)  
module rtl_model (input            ap_clk, ap_rst, ap_ce, ap_start, ap_continue,
                  input [31:0]      a1, b1,
                  output           ap_idle, ap_done, ap_ready,
                  output           z1_ap_vld,
                  output reg [31:0] z1);

   wire ce = ap_ce;
   
   reg [31:0] areg1;
   reg [31:0] breg1;
   reg       dly1, dly2;
   
   always @ (posedge ap_clk)
     if (ap_rst)
       begin
          z1    <= 0;
          areg1 <= 0;
          breg1 <= 0;
          dly1  <= 0;
          dly2  <= 0;     
       end
     else if (ce)
       begin
          z1    <= areg1 + breg1;
          areg1 <= a1;
          breg1 <= b1;
          dly1  <= ap_start;
          dly2  <= dly1;          
       end

   assign z1_ap_vld = dly2;
   assign ap_ready  = dly2;
   assign ap_done   = dly2;
   assign ap_idle   = ~ap_start;
      
endmodule // rtl_model
