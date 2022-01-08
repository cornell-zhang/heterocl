/*===============================================================*/
/*                                                               */
/*                           utils.h                             */
/*                                                               */
/*                       Utility functions                       */
/*                                                               */
/*===============================================================*/

#include <string>
//target device
const std::string TARGET_DEVICE = "xilinx_aws-vu9p-f1-04261818_dynamic_5_0";

void print_usage(char* filename);

void parse_sdaccel_command_line_args(
    int argc,
    char** argv,
    std::string& kernelFile);

