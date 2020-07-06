/*===============================================================*/
/*                                                               */
/*                          CLWorld.h                            */
/*                                                               */
/*          Defines the object class for OpenCL context          */
/*                                                               */
/*===============================================================*/


#ifndef __CLWorld__Harness__
#define __CLWorld__Harness__
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

// standard headers
#include <cstdio>
#include <string>
#include <vector>
// opencl header
#include <CL/cl.h>
// CLKernel and CLMemObj are members of this class
#include "CLKernel.h"
#include "CLMemObj.h"

namespace rosetta
{

  class CLWorld
  {
    
    public:

      // default constructor
      CLWorld();

      // meaningful constructor
      CLWorld(std::string target_device_name, cl_device_type device_type);

      // get the compute device associated with this world
      cl_device_id getDevice();

      // get the compute context associated with this world
      cl_context getContext();

      // get the binary program 
      cl_program getProgram();

      // insert a compute program 
      int addProgram(std::string filename);

      // insert a kernel
      int addKernel(CLKernel &new_kernel);

      // insert a memory object
      int addMemObj(CLMemObj &new_mem_obj);

      // update a memory object (write new value)
      int updateMemObj(int mem_id);

      // read a memory object
      int readMemObj(int mem_id);

      // set memory kernel argument
      int setMemKernelArg(int kernel_id, int pos, int mem_id);
      int setIntKernelArg(int kernel_id, int pos, int data);

      // set constant kernel argument
      template<typename T>
      int setConstKernelArg(int kernel_id, int pos, T& arg)
      {
	// printf("%lu\n", arg);
        printf("Set const arg %d for kernel %d ... ", pos, kernel_id);
    
        int err = kernels[kernel_id].set_const_arg(pos, arg);
        if (err != CL_SUCCESS)
        {
          printf("Error setting kernel argument!\n");
          printf("Error code %d\n", err);
          exit(EXIT_FAILURE);
        }
    
        printf("Done!\n");
    
        return err;
      }

      // run kernels
      int runKernels(bool flush = false);

      // clean up
      void releaseWorld();

    private:

      // OpenCL runtime variables

      // the platform we will use
      cl_platform_id platform;

      // the device we will use
      std::string target_device_name;	// device name
      cl_device_type device_type;       // device type
      cl_device_id device_id;           // device id

      // compute context
      cl_context context;

      // command queue
      cl_command_queue cmd_queue;        

      // binary program for the device
      char* kernel_code;
      cl_program program;

      // kernels
      std::vector<CLKernel> kernels;

      // memory objects
      std::vector<CLMemObj> mem_objs;
      // actual OpenCL memory buffers
      std::vector<cl_mem>   cl_mem_buffers;

      // function to create the OpenCL runtime
      int createWorld();

      // load binary file into memory
      int load_file_to_memory(const char *filename);
  };

}

#endif
