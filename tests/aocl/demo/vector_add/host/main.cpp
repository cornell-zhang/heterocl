#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "CL/opencl.h"

// Platform Layer
cl_platform_id platform = NULL;
cl_device_id device; 
unsigned num_devices = 0;
cl_context context = NULL;

// Runtime Layer

cl_program program = NULL;
cl_command_queue queue;
cl_kernel kernel;

// Buffer Layer
cl_mem input1;
cl_mem input2;
cl_mem output;

void init_opencl();
void init_problem();
void run();
void cleanup();

int main()
{
    init_opencl();
}


void init_opencl()
{
    cl_int error;

    printf("Initializing OpenCL\n")

    error = clGetPlatformIDs(1,&platform,NULL);
    error = clGetDeviceIDs(platform,CL_DEVICE_TYPE_ACCELERATOR,1,&device,NULL);
    context = clCreateContext(NULL,1,&device,NULL,NULL,&error);
    // Create Program
    program = clCreateProgramWithBinary
    // Build Program (without actual function)
    error = clBuildProgram
    queue = clCreateCommandQueue(context,device,0,&error);
    
    // Create Kernel
    kernel = clCreateKernel(program);

    input1 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*count,NULL,&error);
    input2 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*count,NULL,&error);
    output = clCraeteBuffer(context, CL_MEM_WRITE_ONLY,sizeof(float)*count,NULL,&error);

    error = clEnqueueWriteBuffer(queue, input1, CL_TRUE,0,sizeof(float)*count, data_a,0,NULL,NULL);
    error = clEnqueueWriteBuffer(queue, input2, CL_TRUE,0,sizeof(float)*count, data_b,0,NULL,NULL);

    error 
    

}