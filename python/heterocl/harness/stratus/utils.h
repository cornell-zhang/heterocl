#ifndef UTILS
#define UTILS

#include "hw_spec.h"
#include <iostream>
#include <fstream>

std::string printBits(size_t const size, void const * const ptr) {
    unsigned char *b = (unsigned char*) ptr;
    bool byte;
    int i, j;
    std::stringstream ss;

    for (i = size-1; i >= 0; i--) {
        for (j = 7; j >= 0; j--) {
            byte = (b[i] >> j) & 1;
            ss << byte;
        }
    }
    return ss.str();
}

std::string printBufferInBits(unsigned char* buffer, int size) {
    std::stringstream ss;
    for (int i = size-1; i >= 0; i--) {
        ss << printBits(1, &buffer[i]);
        if (i % 16 == 0 && i != size-1)
            ss << "\n";
    }
    return ss.str();
}

// Helper function: Get Opcode string
const char* getOpcodeString(int opcode, bool use_imm) {
// The string name
if (opcode == VTA_ALU_OPCODE_MIN) {
    if (use_imm) {
    return "min imm";
    } else {
    return "min";
    }
} else if (opcode == VTA_ALU_OPCODE_MAX) {
    if (use_imm) {
    return "max imm";
    } else {
    return "max";
    }
} else if (opcode == VTA_ALU_OPCODE_ADD) {
    if (use_imm) {
    return "add imm";
    } else {
    return "add";
    }
} else if (opcode == VTA_ALU_OPCODE_SHR) {
    return "shr";
} else if (opcode == VTA_ALU_OPCODE_CLP) {
    return "clp";
}
else if (opcode == VTA_ALU_OPCODE_MOV) {
    if (use_imm) {
    return "mov imm";
    } else {
    return "mov";
    }
}

return "unknown op";
}

void PrintInsn(const VTAGenericInsn* insn, int insn_count) {
    FILE *pFile;
    pFile = fopen("insn_decoded.txt", "w");
    // Keep tabs on dependence queues
    int l2g_queue = 0;
    int g2l_queue = 0;
    int s2g_queue = 0;
    int g2s_queue = 0;
    // Converter
    union VTAInsn c;
    // Iterate over all instructions
    printf("There are %u instructions\n", insn_count);
    for (int i = 0; i < insn_count; ++i) {
        // Fetch instruction and decode opcode
        c.generic = insn[i];
        cout << printBits(sizeof(c.generic), &c.generic) << endl;
        printf("INSTRUCTION %u: ", i);
        fprintf(pFile, "INSTRUCTION %u: ", i);
        if (c.mem.opcode == VTA_OPCODE_LOAD || c.mem.opcode == VTA_OPCODE_STORE) {
        if (c.mem.x_size == 0) {
            if (c.mem.opcode == VTA_OPCODE_STORE) {
                printf("NOP-STORE-STAGE\n");
                fprintf(pFile, "NOP-STORE-STAGE\n");
            } else {
                printf("NOP-MEMORY-STAGE or NOP-COMPUTE-STAGE\n");
                fprintf(pFile, "NOP-MEMORY-STAGE or NOP-COMPUTE-STAGE\n");
            }
                printf("\tdep - pop prev: %d, pop next: %d, push prev: %d, push next: %d\n",
                    static_cast<int>(c.mem.pop_prev_dep), static_cast<int>(c.mem.pop_next_dep),
                    static_cast<int>(c.mem.push_prev_dep), static_cast<int>(c.mem.push_next_dep));
            fprintf(pFile, "\tdep - pop prev: %d, pop next: %d, push prev: %d, push next: %d\n",
                    static_cast<int>(c.mem.pop_prev_dep), static_cast<int>(c.mem.pop_next_dep),
                    static_cast<int>(c.mem.push_prev_dep), static_cast<int>(c.mem.push_next_dep));
            // Count status in queues
            if (c.mem.opcode == VTA_OPCODE_STORE) {
            if (c.mem.pop_prev_dep) g2s_queue--;
            if (c.mem.push_prev_dep) s2g_queue++;
            } else if (c.mem.opcode == VTA_OPCODE_LOAD &&
                        (c.mem.memory_type == VTA_MEM_ID_INP || c.mem.memory_type == VTA_MEM_ID_WGT)) {
            if (c.mem.pop_next_dep) g2l_queue--;
            if (c.mem.push_next_dep) l2g_queue++;
            } else {
            if (c.mem.pop_prev_dep) l2g_queue--;
            if (c.mem.push_prev_dep) g2l_queue++;
            if (c.mem.pop_next_dep) s2g_queue--;
            if (c.mem.push_next_dep) g2s_queue++;
            }
            printf("\tl2g_queue = %d, g2l_queue = %d\n", l2g_queue, g2l_queue);
            fprintf(pFile, "\tl2g_queue = %d, g2l_queue = %d\n", l2g_queue, g2l_queue);
            printf("\ts2g_queue = %d, g2s_queue = %d\n", s2g_queue, g2s_queue);
            fprintf(pFile, "\ts2g_queue = %d, g2s_queue = %d\n", s2g_queue, g2s_queue);
            continue;
        }
        // Print instruction field information
        if (c.mem.opcode == VTA_OPCODE_LOAD) {
            printf("LOAD ");
            fprintf(pFile, "LOAD ");
            if (c.mem.memory_type == VTA_MEM_ID_UOP) {
                printf("UOP\n");
                fprintf(pFile, "UOP\n");
            }
            if (c.mem.memory_type == VTA_MEM_ID_WGT) {
                printf("WGT\n");
                fprintf(pFile, "WGT\n");
            }
            if (c.mem.memory_type == VTA_MEM_ID_INP) {
                printf("INP\n");
                fprintf(pFile, "INP\n");
            }
            if (c.mem.memory_type == VTA_MEM_ID_ACC) {
                printf("ACC\n");
                fprintf(pFile, "ACC\n");
            }
        }
        if (c.mem.opcode == VTA_OPCODE_STORE) {
            printf("STORE:\n");
            fprintf(pFile, "STORE:\n");
        }
        printf("\tdep - pop prev: %d, pop next: %d, push prev: %d, push next: %d\n",
                static_cast<int>(c.mem.pop_prev_dep), static_cast<int>(c.mem.pop_next_dep),
                static_cast<int>(c.mem.push_prev_dep), static_cast<int>(c.mem.push_next_dep));
        printf("\tDRAM: 0x%08x, SRAM:0x%04x\n", static_cast<int>(c.mem.dram_base),
                static_cast<int>(c.mem.sram_base));
        printf("\ty: size=%d, pad=[%d, %d]\n", static_cast<int>(c.mem.y_size),
                static_cast<int>(c.mem.y_pad_0), static_cast<int>(c.mem.y_pad_1));
        printf("\tx: size=%d, stride=%d, pad=[%d, %d]\n", static_cast<int>(c.mem.x_size),
                static_cast<int>(c.mem.x_stride), static_cast<int>(c.mem.x_pad_0),
                static_cast<int>(c.mem.x_pad_1));
        fprintf(pFile, "\tdep - pop prev: %d, pop next: %d, push prev: %d, push next: %d\n",
                static_cast<int>(c.mem.pop_prev_dep), static_cast<int>(c.mem.pop_next_dep),
                static_cast<int>(c.mem.push_prev_dep), static_cast<int>(c.mem.push_next_dep));
        fprintf(pFile, "\tDRAM: 0x%08x, SRAM:0x%04x\n", static_cast<int>(c.mem.dram_base),
                static_cast<int>(c.mem.sram_base));
        fprintf(pFile, "\ty: size=%d, pad=[%d, %d]\n", static_cast<int>(c.mem.y_size),
                static_cast<int>(c.mem.y_pad_0), static_cast<int>(c.mem.y_pad_1));
        fprintf(pFile, "\tx: size=%d, stride=%d, pad=[%d, %d]\n", static_cast<int>(c.mem.x_size),
                static_cast<int>(c.mem.x_stride), static_cast<int>(c.mem.x_pad_0),
                static_cast<int>(c.mem.x_pad_1));

        } else if (c.mem.opcode == VTA_OPCODE_GEMM) {
        // Print instruction field information
        printf("GEMM\n");

        printf("\tdep - pop prev: %d, pop next: %d, push prev: %d, push next: %d\n",
                static_cast<int>(c.mem.pop_prev_dep), static_cast<int>(c.mem.pop_next_dep),
                static_cast<int>(c.mem.push_prev_dep), static_cast<int>(c.mem.push_next_dep));
        printf("\treset_out: %d\n", static_cast<int>(c.gemm.reset_reg));
        printf("\trange (%d, %d)\n", static_cast<int>(c.gemm.uop_bgn),
                static_cast<int>(c.gemm.uop_end));
        printf("\touter loop - iter: %d, wgt: %d, inp: %d, acc: %d\n",
                static_cast<int>(c.gemm.iter_out), static_cast<int>(c.gemm.wgt_factor_out),
                static_cast<int>(c.gemm.src_factor_out), static_cast<int>(c.gemm.dst_factor_out));
        printf("\tinner loop - iter: %d, wgt: %d, inp: %d, acc: %d\n",
                static_cast<int>(c.gemm.iter_in), static_cast<int>(c.gemm.wgt_factor_in),
                static_cast<int>(c.gemm.src_factor_in), static_cast<int>(c.gemm.dst_factor_in));
        fprintf(pFile, "GEMM\n");

        fprintf(pFile, "\tdep - pop prev: %d, pop next: %d, push prev: %d, push next: %d\n",
                static_cast<int>(c.mem.pop_prev_dep), static_cast<int>(c.mem.pop_next_dep),
                static_cast<int>(c.mem.push_prev_dep), static_cast<int>(c.mem.push_next_dep));
        fprintf(pFile, "\treset_out: %d\n", static_cast<int>(c.gemm.reset_reg));
        fprintf(pFile, "\trange (%d, %d)\n", static_cast<int>(c.gemm.uop_bgn),
                static_cast<int>(c.gemm.uop_end));
        fprintf(pFile, "\touter loop - iter: %d, wgt: %d, inp: %d, acc: %d\n",
                static_cast<int>(c.gemm.iter_out), static_cast<int>(c.gemm.wgt_factor_out),
                static_cast<int>(c.gemm.src_factor_out), static_cast<int>(c.gemm.dst_factor_out));
        fprintf(pFile, "\tinner loop - iter: %d, wgt: %d, inp: %d, acc: %d\n",
                static_cast<int>(c.gemm.iter_in), static_cast<int>(c.gemm.wgt_factor_in),
                static_cast<int>(c.gemm.src_factor_in), static_cast<int>(c.gemm.dst_factor_in));

        } else if (c.mem.opcode == VTA_OPCODE_ALU) {
        // Print instruction field information
        printf("ALU - %s\n", getOpcodeString(c.alu.alu_opcode, c.alu.use_imm));
        printf("\tdep - pop prev: %d, pop next: %d, push prev: %d, push next: %d\n",
                static_cast<int>(c.mem.pop_prev_dep), static_cast<int>(c.mem.pop_next_dep),
                static_cast<int>(c.mem.push_prev_dep), static_cast<int>(c.mem.push_next_dep));
        printf("\treset_out: %d\n", static_cast<int>(c.alu.reset_reg));
        printf("\trange (%d, %d)\n", static_cast<int>(c.alu.uop_bgn),
                static_cast<int>(c.alu.uop_end));
        printf("\touter loop - iter: %d, dst: %d, src: %d\n", static_cast<int>(c.alu.iter_out),
                static_cast<int>(c.alu.dst_factor_out), static_cast<int>(c.alu.src_factor_out));
        printf("\tinner loop - iter: %d, dst: %d, src: %d\n", static_cast<int>(c.alu.iter_in),
                static_cast<int>(c.alu.dst_factor_in), static_cast<int>(c.alu.src_factor_in));

        fprintf(pFile, "ALU - %s\n", getOpcodeString(c.alu.alu_opcode, c.alu.use_imm));
        fprintf(pFile, "\tdep - pop prev: %d, pop next: %d, push prev: %d, push next: %d\n",
                static_cast<int>(c.mem.pop_prev_dep), static_cast<int>(c.mem.pop_next_dep),
                static_cast<int>(c.mem.push_prev_dep), static_cast<int>(c.mem.push_next_dep));
        fprintf(pFile, "\treset_out: %d\n", static_cast<int>(c.alu.reset_reg));
        fprintf(pFile, "\trange (%d, %d)\n", static_cast<int>(c.alu.uop_bgn),
                static_cast<int>(c.alu.uop_end));
        fprintf(pFile, "\touter loop - iter: %d, dst: %d, src: %d\n", static_cast<int>(c.alu.iter_out),
                static_cast<int>(c.alu.dst_factor_out), static_cast<int>(c.alu.src_factor_out));
        fprintf(pFile, "\tinner loop - iter: %d, dst: %d, src: %d\n", static_cast<int>(c.alu.iter_in),
                static_cast<int>(c.alu.dst_factor_in), static_cast<int>(c.alu.src_factor_in));
        
        } else if (c.mem.opcode == VTA_OPCODE_FINISH) {
            printf("FINISH\n");
            fprintf(pFile, "FINISH\n");
        }

        // Count status in queues
        if (c.mem.opcode == VTA_OPCODE_LOAD || c.mem.opcode == VTA_OPCODE_STORE) {
        if (c.mem.opcode == VTA_OPCODE_STORE) {
            if (c.mem.pop_prev_dep) g2s_queue--;
            if (c.mem.push_prev_dep) s2g_queue++;
        } else if (c.mem.opcode == VTA_OPCODE_LOAD &&
                    (c.mem.memory_type == VTA_MEM_ID_INP || c.mem.memory_type == VTA_MEM_ID_WGT)) {
            if (c.mem.pop_next_dep) g2l_queue--;
            if (c.mem.push_next_dep) l2g_queue++;
        } else {
            if (c.mem.pop_prev_dep) l2g_queue--;
            if (c.mem.push_prev_dep) g2l_queue++;
            if (c.mem.pop_next_dep) s2g_queue--;
            if (c.mem.push_next_dep) g2s_queue++;
        }
        } else if (c.mem.opcode == VTA_OPCODE_GEMM || c.mem.opcode == VTA_OPCODE_ALU) {
        // Print instruction field information
        if (c.gemm.pop_prev_dep) l2g_queue--;
        if (c.gemm.push_prev_dep) g2l_queue++;
        if (c.gemm.pop_next_dep) s2g_queue--;
        if (c.gemm.push_next_dep) g2s_queue++;
        }
        printf("\tl2g_queue = %d, g2l_queue = %d\n", l2g_queue, g2l_queue);
        fprintf(pFile, "\tl2g_queue = %d, g2l_queue = %d\n", l2g_queue, g2l_queue);
        printf("\ts2g_queue = %d, g2s_queue = %d\n", s2g_queue, g2s_queue);
        fprintf(pFile, "\ts2g_queue = %d, g2s_queue = %d\n", s2g_queue, g2s_queue);
    }
    fclose(pFile);
}


void log(std::string name, std::string msg) {
    cout << "========> [" << name << "] " << msg << endl;
}



#endif // UTILS