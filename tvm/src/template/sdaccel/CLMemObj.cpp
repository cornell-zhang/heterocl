/*===============================================================*/
/*                                                               */
/*                        CLMemObj.cpp                           */
/*                                                               */
/*       Implements the member functions of CLMemObj class       */
/*                                                               */
/*===============================================================*/


#include "CLMemObj.h"

namespace rosetta
{
  // default constructor, initializes everything to 0
  CLMemObj::CLMemObj() 
  {
    this->mem_data = nullptr;
    this->elt_size = 0;
    this->length   = 0;
    this->flags    = 0;
    this->bank     = nullptr;
  }
  
  // meaningful constructor, initialize data info constants
  CLMemObj::CLMemObj(void *mem_data, int elt_size, int length, cl_mem_flags flags, cl_mem_ext_ptr_t* xil_ext ) 
  {
    this->mem_data = mem_data;
    this->elt_size = elt_size;
    this->length   = length;
    this->flags    = flags;
    // can use Xilinx mem extensions to specify DDR bank
    if (xil_ext != nullptr)
    {
      this->bank = new cl_mem_ext_ptr_t;
      this->bank->flags = xil_ext->flags;
      this->bank->obj = xil_ext->obj;
      this->bank->param = 0;
    }
    else
      this->bank = nullptr;
  }
  
  // return the pointer to data
  void * CLMemObj::get_data()  { return mem_data; }
  
  // get size of each element
  int CLMemObj::get_element_size() { return elt_size; }
  
  // get the number of elements in the buffer
  int CLMemObj::get_length() { return length; }
  
  // get OpenCL memory flags
  cl_mem_flags CLMemObj::get_flags() { return flags; }

  // get xilinx memory extension pointer
  cl_mem_ext_ptr_t* CLMemObj::get_xil_ext_ptr() { return bank; }
}
