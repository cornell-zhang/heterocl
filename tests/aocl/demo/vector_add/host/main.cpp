#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "CL/opencl.h"

// Platform Layer
cl_platform_id platform = NULL;
cl_device_id device; 
unsigned num_devices = 0;
cl_context context = NULL;



cl_program program = NULL;

bool init_opencl();
void init_problem();
void run();
void cleanup();

int main()
{

}


bool init_opencl()
{
    cl_int error;

    printf("Initializing OpenCL\n")

    error = clGetPlatformIDs(1,&platform,NULL);
    error = clGetDeviceIDs(platform,CL_DEVICE_TYPE_ACCELERATOR,1,&device,NULL);
    context = clCreateContext(NULL,1,&device,NULL,NULL,&error);

    
}