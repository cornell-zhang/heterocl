/*===============================================================*/
/*                                                               */
/*                         CLWorld.cpp                           */
/*                                                               */
/*             Implementation of the CLWorld class               */
/*                                                               */
/*===============================================================*/

#include "CLWorld.h"

namespace rosetta
{
  // default constructor
  // make sure it does something meaningful
  CLWorld::CLWorld()
  {
    // default: run on alpha data 7v3 board
    this->target_device_name = "xilinx:adm-pcie-7v3:1ddr:3.0";
    this->device_type = CL_DEVICE_TYPE_ACCELERATOR;

    // configure the OpenCL runtime
    createWorld();
  }

  // meaningful constructor
  // user specifies device
  CLWorld::CLWorld(std::string target_device_name, cl_device_type device_type)
  {
    this->target_device_name = target_device_name;
    this->device_type = device_type;
    createWorld();
  }

  // get the compute device
  cl_device_id CLWorld::getDevice()
  {
    return this->device_id;
  }

  // get context
  cl_context CLWorld::getContext()
  {
    return this->context;
  }

  // get compute program
  cl_program CLWorld::getProgram()
  {
    return this->program;
  }

  // insert a new memory object
  int CLWorld::addMemObj(CLMemObj &new_mem_obj)
  {
    int err;

    printf("Adding memory object into the world ... ");

    // first push the CLMemObj object into our vector
    mem_objs.push_back(new_mem_obj);

    // then create the actual cl_mem buffer, push it into another vector
    cl_mem buf;

    buf = clCreateBuffer(context, new_mem_obj.flags, new_mem_obj.elt_size * new_mem_obj.length, new_mem_obj.bank, &err);
    if (err != CL_SUCCESS)
    {
      printf("Error creating buffer for memory object %d!\n", mem_objs.size()-1);
      printf("Error Code %d\n", err);
      exit(EXIT_FAILURE);
    }

    cl_mem_buffers.push_back(buf);

    // write the buffer onto the device if needed
    if ((new_mem_obj.flags != CL_MEM_WRITE_ONLY) && (new_mem_obj.mem_data != nullptr))
    {
      err = clEnqueueWriteBuffer(cmd_queue, buf, true, 0, new_mem_obj.elt_size * new_mem_obj.length, 
                                 new_mem_obj.mem_data, 0, NULL, NULL);
      if (err != CL_SUCCESS)
      {
        printf("Error writing buffer %d onto the device!\n", mem_objs.size()-1);
        printf("Error Code %d\n", err);
        exit(EXIT_FAILURE);
      }
    }

    printf("Done!\n");

    return (mem_objs.size() - 1);
  }

  int CLWorld::updateMemObj(int mem_idx)
  {
    printf("Updating mem object %d ... ", mem_idx);

    // write the buffer onto the device if needed
    if (mem_objs[mem_idx].flags != CL_MEM_WRITE_ONLY)
    {
      int err = clEnqueueWriteBuffer(cmd_queue, cl_mem_buffers[mem_idx], true, 0, 
                                     mem_objs[mem_idx].elt_size * mem_objs[mem_idx].length, 
                                     mem_objs[mem_idx].mem_data, 0, NULL, NULL);
      if (err != CL_SUCCESS)
      {
        printf("Error writing buffer %d onto the device!\n", mem_idx);
        printf("Error Code %d\n", err);
        exit(EXIT_FAILURE);
      }
    }
    else
      printf("Buffer %d is write_only! Not updating it ... \n", mem_idx);
    
    return EXIT_SUCCESS;
  }
   
  int CLWorld::readMemObj(int mem_idx)
  {
    printf("Reading mem object %d into host buffers ... ", mem_idx);

    int err = clEnqueueReadBuffer(cmd_queue, cl_mem_buffers[mem_idx], true, 0,
                                  mem_objs[mem_idx].elt_size * mem_objs[mem_idx].length, 
				  mem_objs[mem_idx].mem_data, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
      printf("Error reading kernel buffer %d!\n", mem_idx);
      printf("Error code %d\n", err);
      exit(EXIT_FAILURE);
    }

    printf("Done!\n");

    return err;
  }
     

  // create compute program from a file
  // return error code
  int CLWorld::addProgram(std::string filename)
  {
    printf("Adding binary program into the world ... ");

    // load the file
    size_t code_size = (size_t) load_file_to_memory(filename.c_str());

    // start to compile
    int err;
    cl_int create_binary_status;

    // Create the compute program from the source buffer
    program = clCreateProgramWithBinary(context, 1, &device_id, (const size_t *) &code_size, 
                                        (const unsigned char **) &kernel_code, &create_binary_status, &err);
    if (!program)
    {
      printf("Error: Failed to create compute program!\n");
      printf("Error Code %d\n", err);
      exit(EXIT_FAILURE);
    }
 
    // Build the program executable
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
      size_t len;
      char buffer[2048];
 
      printf("Error: Failed to build program executable!\n");
      printf("Error Code %d\n", err);
      clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
      printf("%s\n", buffer);
      exit(EXIT_FAILURE);
    }

    printf("Done!\n");

    return err;
  }

  // insert a kernel into the world
  // return the position of the kernel in the vector
  int CLWorld::addKernel(CLKernel &new_kernel)
  {
    printf("Adding kernel %s into the world ... ", new_kernel.get_name().c_str());

    kernels.push_back(new_kernel);

    printf("Done!\n");

    return (kernels.size() - 1);
  }

  // methods to set kernel arguments
  // memory argument
  int CLWorld::setMemKernelArg(int kernel_id, int pos, int arg_id)
  {
    printf("Set mem arg %d for kernel %d with mem object %d ... ", pos, kernel_id, arg_id);

    int err = kernels[kernel_id].set_mem_arg(pos, cl_mem_buffers[arg_id]);
    if (err != CL_SUCCESS)
    {
      printf("Error setting kernel argument!\n");
      printf("Error code %d\n", err);
      exit(EXIT_FAILURE);
    }

    printf("Done!\n");

    return err;
  }
   
  // run all kernels
  // return error code
  int CLWorld::runKernels(bool flush)
  {
    printf("Start kernel execution ... ");

    int err;

    // wait for previous write buffer tasks to finish
    printf("Waiting for queue... \n");
    clFinish(cmd_queue);

    // enqueue all the kernels
    // temporarily we assume kernels won't have any dependency between them
    // or the dependency is handled inside kernels (such as pipes, etc. )
    for (int i = 0; i < kernels.size(); i ++ )
    {
      printf("Start kernel %d!\n", i);
      err = clEnqueueNDRangeKernel(cmd_queue, kernels[i].kernel, 3, NULL, kernels[i].global_size, kernels[i].local_size, 
                                   0, NULL, NULL);
      if (err != CL_SUCCESS)
      {
        printf("Error enqueuing kernel %d!\n", i);
	printf("Error Code %d\n", err);
	exit(EXIT_FAILURE);
      }
    }

    // wait for them to finish
    printf("Waiting for kernels ... \n");
    clFinish(cmd_queue);

    // remove all of them from the vector
    // so that this function can be called multiple times
    // at a cost that kernels won't be released automatically
    if (flush)
    {
      int total_size = kernels.size();
      for (int i = 0; i < total_size; i ++ )
        kernels.pop_back();
    }

    printf("Done!\n");

    return err;
  }

  // create runtime environment
  int CLWorld::createWorld()
  {
    printf("Initializing OpenCL runtime environment ... ");

    int err;

    // scan the machine for available OpenCL platforms
    cl_uint platform_cnt;
    cl_platform_id platforms[16];
    err = clGetPlatformIDs(16, platforms, &platform_cnt);
    if (err != CL_SUCCESS)
    {
      printf("Error: Failed to find an OpenCL platform!\n");
      printf("Error Code %d\n", err);
      printf("Test failed\n");
      exit(EXIT_FAILURE);
    }
    printf("INFO: Found %d platforms\n", platform_cnt);


    // find the target device
    char device_name[1024];
    cl_device_id devices[16];
    cl_uint device_cnt;
    bool found_device = false;
    // scan all platforms
    for (int p = 0; (p < platform_cnt) & (!found_device); p ++ )
    {
      err = clGetDeviceIDs(platforms[p], this->device_type, 16, devices, &device_cnt);
      if (err != CL_SUCCESS)
      {
        printf("Error: Failed to create a device group for platform %d!\n", p);
        printf("Error Code %d\n", err);
        printf("Test failed\n");
        exit(EXIT_FAILURE);
      }
      // iterate through all devices on the platform
      for (int d = 0; (d < device_cnt) & (!found_device); d ++ )
      {
        err = clGetDeviceInfo(devices[d], CL_DEVICE_NAME, 1024, device_name, 0);
        if (err != CL_SUCCESS) 
	{
	  printf("Error: Failed to get device name for device %d on platform %d!\n", d, p);
	  printf("Error Code %d\n", err);
	  printf("Test failed\n");
	  exit(EXIT_FAILURE);
	}

        if (std::string(device_name) == this->target_device_name)
	{
	  this->platform = platforms[p];
	  this->device_id = devices[d];
          found_device = true;
	  printf("Selected device %d on platform %d as target device!\n", d, p);
	}
      }
    }

    if (!found_device)
    {
      printf("Error: Target device %s is not found!\n", (this->target_device_name).c_str());
      exit(EXIT_FAILURE);
    }

    // create context and command queue
    this->context = clCreateContext(0, 1, &(this->device_id), 0, 0, &err);
    if (!(this->context))
    {
      printf("Error: Failed to create a compute context!\n");
      printf("Error Code %d\n", err);
      exit(EXIT_FAILURE);
    }
    this->cmd_queue = clCreateCommandQueue(this->context, this->device_id, 
                                           CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
					   &err);
    if (!(this->cmd_queue))
    {
      printf("Error: Failed to create a command queue!\n");
      printf("Error Code %d\n", err);
      exit(EXIT_FAILURE);
    }

    printf("Done!\n");

    return err;
  }

  // read kernel binary file into memory
  int CLWorld::load_file_to_memory(const char *filename) 
  {
    int size = 0;
    FILE *f = fopen(filename, "rb");
    if (f == NULL)
    {
      kernel_code = NULL;
      printf("Can not open kernel file!\n");
      exit(-1);
    }
    fseek(f, 0, SEEK_END);
    size = ftell(f);
    printf("Size of the file is %ld\n", size);
    fseek(f, 0, SEEK_SET);
    kernel_code = new char[size+1];
    if ((unsigned int) size != fread(kernel_code, sizeof(char), size, f))
    {
      delete []kernel_code;
      printf("Reading kernel failed!\n");
      exit(-2);
    }
    fclose(f);
    (kernel_code)[size] = 0;
    return size;
  }


  // release all runtime constructs
  void CLWorld::releaseWorld()
  {
    printf("Cleaning up OpenCL opjects ... ");

    // release memory objects
    for (int i = 0; i < cl_mem_buffers.size(); i ++ )
      clReleaseMemObject(cl_mem_buffers[i]);

    // release program
    delete []kernel_code;
    clReleaseProgram(program);

    // release kernels
    for (int i = 0; i < kernels.size(); i ++ )
      kernels[i].releaseKernel();

    // release device and context
    clReleaseCommandQueue(cmd_queue);
    clReleaseContext(context);

    printf("Done!\n");
  }

}




