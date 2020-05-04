/*===============================================================*/
/*                                                               */
/*                         CLMemObj.h                            */
/*                                                               */
/*     Defines the object class for an OpenCL memory buffer      */
/*                                                               */
/*===============================================================*/


#ifndef __CLMemObj__Harness__
#define __CLMemObj__Harness__

// standard header for command line output
#include <cstdio>
// opencl header
#include <CL/cl.h>
// xilinx opencl extension header
#include <CL/cl_ext.h>

namespace rosetta
{
  // wrapper class around cl_mem
  class CLMemObj 
  {
 
    friend class CLWorld;

    public:
  
      // default constructor
      CLMemObj ();
      // a meaningful constructor
      CLMemObj (void* mem_data, int elt_size, int length, cl_mem_flags flags, cl_mem_ext_ptr_t* xil_ext = nullptr);
  
      // get information about the buffer
      void* get_data();
      int get_element_size();
      int get_length();
      cl_mem_flags get_flags();
      cl_mem_ext_ptr_t* get_xil_ext_ptr();
 
    private:
  
      // pointer to data
      void *mem_data;
      // size of each element
      int elt_size;
      // number of elements
      int length;
      // OpenCL memory flag
      cl_mem_flags flags;
      // Xilinx extension describing bank assignment
      cl_mem_ext_ptr_t* bank;
  };
}

#endif /* defined(__CLMemObj__Harness__) */
