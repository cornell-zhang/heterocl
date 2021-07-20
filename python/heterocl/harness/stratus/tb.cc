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

#include "tb.h"
#include <esc.h>		// for the latency logging functions
#include <string>
#include <iostream>
#include "hw_spec.h"
#include "utils.h"

// The source thread reads data from a file and sends it to the DUT
void tb::source()
{
    log("TB", "source thread starts");
    instr_count.reset();
    instr_phy_addr.reset();
    rst.write(0);
    wait(2);
    rst.write(1);
    wait();
    
    sc_uint<32> instr_count_ = 0;
    sc_uint<32> instr_phy_addr_ = 0; 

    // Read instruction count and base address
    std::ifstream input("./DRAM/insn.txt");
    std::string instr_count_str, instr_base_str;
    std::getline(input, instr_base_str);
    std::getline(input, instr_count_str);
    instr_count_ = std::stoi(instr_count_str);
    instr_phy_addr_ = std::stoi(instr_base_str);
    
    // Read instructions
    FILE* insn_p;
    insn_p = fopen("./DRAM/DRAM.bin", "rb");
    if (insn_p == NULL) {
        cout << "Can't open DRAM file, check if it exists." << endl;
        exit(0);
    }

    fseek(insn_p, 0, SEEK_END);
    len = ftell(insn_p); 
    buff = new unsigned char[len];
    rewind(insn_p);
    fread(buff, 1, len, insn_p);
    fclose(insn_p);
    log("TB", "Load DRAM file done.");

    // Write value from input buffer to DRAM array 
    for (int i = 0; i < len; i++)
        DRAM[i] = (sc_uint<8>) buff[i];


    // Decode instructions
    VTAGenericInsn* vta_insns = static_cast<VTAGenericInsn *>(malloc(sizeof(VTAGenericInsn) * instr_count_));

    int insn_idx = 0;
    for (int offset = 0; offset < instr_count_ * 16; offset += 16) {
        int i = instr_phy_addr_ + offset;
        VTAGenericInsn insn = {};
        insn.opcode = DRAM[i] & 0b00000111; // byte[0][0:3]
        insn.pop_prev_dep = (DRAM[i] >> 3) & 0b00000001; // byte[0][3:4]
        insn.pop_next_dep = (DRAM[i] >> 4) & 0b00000001; // byte[0][4:5]
        insn.push_prev_dep = (DRAM[i] >> 5) & 0b00000001; // byte[0][5:6]
        insn.push_next_dep = (DRAM[i] >> 6) & 0b00000001; // byte[0][6:7]
        // { byte[0][7:] : byte[1:8] }
        bool byte_0_7 = (DRAM[i] >> 7) & 0b00000001;
        insn.pad_0 =  ((uint64_t)byte_0_7)                | ((uint64_t)DRAM[i+1] << (8*0 + 1))
                    | ((uint64_t)DRAM[i+2] << (8*1 + 1))  | ((uint64_t)DRAM[i+3] << (8*2 + 1))
                    | ((uint64_t)DRAM[i+4] << (8*3 + 1))  | ((uint64_t)DRAM[i+5] << (8*4 + 1))
                    | ((uint64_t)DRAM[i+6] << (8*5 + 1))  | ((uint64_t)DRAM[i+7] << (8*6 + 1));
        // byte[8:15]
        insn.pad_1 = ((uint64_t)DRAM[i+8] << 8*0)    | ((uint64_t)DRAM[i+9] << 8*1)
                    | ((uint64_t)DRAM[i+10] << 8*2)  | ((uint64_t)DRAM[i+11] << 8*3)
                    | ((uint64_t)DRAM[i+12] << 8*4)  | ((uint64_t)DRAM[i+13] << 8*5)
                    | ((uint64_t)DRAM[i+14] << 8*6)  | ((uint64_t)DRAM[i+15] << 8*7);
        vta_insns[insn_idx] = insn;
        insn_idx += 1;
    }

    PrintInsn(vta_insns, instr_count_);

    cout << "simulation starts..." << endl;
  
    // Put instr_count and instr_phy_address to P2P Port
    instr_phy_addr.put(instr_phy_addr_);
    instr_count.put(instr_count_);

    //wait(100000);
    //cout << "TB source timed out" << endl;
    //esc_stop();
}



// The sink thread reads all the expected values from the design
void tb::sink()
{
    log("TB", "sink thread starts");
    wait(); // wait for reset signal
    log("TB", "sink thread wait done");
    do {wait();} while (!finish.read());

    // write DRAM to result file
    FILE* result_p = fopen("./DRAM/result.bin", "wb");
    if (result_p == NULL) {
        cout << "Can't open result binary file" << endl;
        exit(0);
    }

    for (int i = 0; i < len; i++)
        buff[i] = (unsigned char)DRAM[i];

    fwrite(buff, 1, len, result_p);
    fclose(result_p);
    log("TB", "DRAM results written to file.");
    

    cout << "Normal turn off." << endl;
    esc_stop();
}
