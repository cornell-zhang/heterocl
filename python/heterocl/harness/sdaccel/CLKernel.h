/*===============================================================*/
/*                                                               */
/*                         CLKernel.h                            */
/*                                                               */
/*          Defines the object class for an OpenCL kernel        */
/*                                                               */
/*===============================================================*/


#ifndef __CLKernel__Harness__
#define __CLKernel__Harness__

// standard headers
#include <cstdio>
#include <vector>
#include <string>
// opencl header
#include <CL/cl.h>
// CLMemObj is a member of this class
#include "CLMemObj.h"

namespace rosetta
{
  
  // wrapper class around an OpenCL kernel
  class CLKernel 
  {

    friend class CLWorld;

    public:

      // constructor
      // compiles the kernel
      CLKernel(cl_context context, cl_program program, std::string kernel_name, cl_device_id device_id);

      // set global/local work group size
      void set_global(int global_work_size[3]);
      void set_local(int local_work_size[3]);

      // get kernel name
      std::string get_name();

    protected:

      int set_int_arg(int id, int data)
      {
        int err;
        err = clSetKernelArg(this->kernel, id, sizeof(data), &data);
      }
      // set cl_mem argument
      int set_mem_arg(int id, cl_mem mem_obj)
      {
        int err;
        err = clSetKernelArg(this->kernel, id, sizeof(mem_obj), &mem_obj);
        if (err != CL_SUCCESS)
        {
          printf("Error: Failed to set kernel argument %d for kernel %s!\n", id, (this->kernel_name).c_str());
          printf("Error Code %d\n", err);
          return EXIT_FAILURE;
        }

        return err;
      }

      // set memory arguments for this kernel
      template<typename T>
      int set_const_arg(int id, T& mem_obj)
      {
        int err;
	// printf("%d\n", mem_obj);
        err = clSetKernelArg(this->kernel, id, sizeof(mem_obj), &mem_obj);
	printf("****************\n");
	printf("%d\n", err);
        if (err != CL_SUCCESS)
        {
          printf("Error: Failed to set kernel argument %d for kernel %s!\n", id, (this->kernel_name).c_str());
          printf("Error Code %d\n", err);
          return EXIT_FAILURE;
        }

        return err;
      }

      void releaseKernel();

    private:

      // global and local work group size
      size_t global_size[3];
      size_t local_size[3];

      // kernel information and objects
      std::string kernel_name;
      cl_device_id device_id;		// target device id
      cl_kernel kernel;                 // compute kernel

  };

}
#endif /* defined(__CLKernel__Harness__) */
