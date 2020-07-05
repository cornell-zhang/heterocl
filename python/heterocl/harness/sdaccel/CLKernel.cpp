/*===============================================================*/
/*                                                               */
/*                         CLKernel.cpp                          */
/*                                                               */
/*          Defines the object class for an OpenCL kernel        */
/*                                                               */
/*===============================================================*/

#include "CLKernel.h"
#include <stdlib.h>

namespace rosetta
{
  // initialize the kernel from binary file
  CLKernel::CLKernel(cl_context context, cl_program program, std::string kernel_name, cl_device_id device_id) 
  {
    printf("Creating kernel %s ... ", kernel_name.c_str());

    int err;

    // set the name and device ID
    this->device_id = device_id;
    this->kernel_name = kernel_name;

    // Create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, kernel_name.c_str(), &err);
    if (!kernel || err != CL_SUCCESS)
    {
      printf("Error: Failed to create compute kernel!\n");
      printf("Error Code %d\n", err);
      exit(EXIT_FAILURE);
    }

    printf("Done!\n");  
  }
 
  void CLKernel::set_global(int global_work_size[3]) 
  {
    printf("Set global work size of kernel %s to [%d, %d, %d]\n", kernel_name.c_str(), 
           global_work_size[0], global_work_size[1], global_work_size[2]);

    for (int i = 0; i < 3; i ++ )
      this->global_size[i] = global_work_size[i];
  }
  
  void CLKernel::set_local(int local_work_size[3]) 
  {
    printf("Set local work size of kernel %s to [%d, %d, %d]\n", kernel_name.c_str(), 
           local_work_size[0], local_work_size[1], local_work_size[2]);

    for (int i = 0; i < 3; i ++ )
      this->local_size[i] = local_work_size[i];
  }

  std::string CLKernel::get_name()
  {
    return this->kernel_name;
  }

  void CLKernel::releaseKernel()
  {
    printf("Release kernel %s ... ", kernel_name.c_str());
    // release kernel
    clReleaseKernel(kernel);
    printf("Done!\n");
  }
}
