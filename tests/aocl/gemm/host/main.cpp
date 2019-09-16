#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include "CL/opencl.h"
#endif

cl_int error;
// Platform Layer
cl_platform_id platform = NULL;
cl_device_id device; 
//unsigned num_devices = 0;
cl_context context = NULL;

// Runtime Layer
cl_program program = NULL;
cl_command_queue queue;
cl_kernel kernel;

// Buffer Layer
cl_mem input1;
cl_mem input2;
cl_mem output;
// length of array
size_t m = 10;
size_t n = 10;
size_t k = 10;
// host data
int data_a[m*k];
int data_b[k*n];
int results[m*n];


void init_opencl();
void init_problem();
void run();
void cleanup();

int main()
{
    init_opencl();
    init_problem();
    run();
    cleanup();
    return 0;
}


void init_opencl()
{

    printf("Initializing OpenCL\n");

    error = clGetPlatformIDs(1,&platform,NULL);
    error = clGetDeviceIDs(platform,CL_DEVICE_TYPE_ACCELERATOR,1,&device,NULL);
    context = clCreateContext(NULL,1,&device,NULL,NULL,&error);

    // Create Program
    size_t binary_length;

    FILE* fp = fopen("bin/gemm.aocx","rb");

    if(fp == NULL)
    {
        printf("Failed to open the AOCX_FILE.\n");
    }
    fseek(fp,0,SEEK_END);
    binary_length = ftell(fp);
    const unsigned char*binary = new unsigned char[binary_length];
    rewind(fp);

    if(fread((void*)binary, binary_length,1,fp)==0)
    {
        printf("Failed to read from the AOCX file.\n");
        fclose(fp);
        exit(1);
    }
    fclose(fp);

    program = clCreateProgramWithBinary(context,1,&device,&binary_length,(const unsigned char**)&binary,&error,NULL);
    
    // Build Program (without actual function)
    error = clBuildProgram(program,1, &device,NULL,NULL,NULL);
    queue = clCreateCommandQueue(context,device,0,&error);
    
    // Create Kernel
    kernel = clCreateKernel(program,"default_function",&error);

    input1 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int)*count,NULL,&error);
    input2 = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int)*count,NULL,&error);
    output = clCreateBuffer(context, CL_MEM_WRITE_ONLY,sizeof(int)*count,NULL,&error);
}

void init_problem()
{
    // generate the input
    srand(time(NULL));
    for(int i=0;i<m*k;i++)
    {
        data_a[i]=rand()%10;
    }
    printf("data_a: \n");
    for(int i=0;i<m;i++)
    {
        for(int j=0;j<k;j++)
        {
            printf("%d ",data_a[i*k+j]);
        }
        printf("\n");
    }

    for(int i=0;i<k*n;i++)
    {
        data_b[i]=rand()%10;
    }
    
    printf("data_b: \n");
    for(int i=0;i<k;i++)
    {
        for(int j=0;j<n;j++)
        {
            printf("%d ",data_b[i*n+j]);
        }
        printf("\n");
    }
}

void run()
{
    error = clEnqueueWriteBuffer(queue, input1, CL_TRUE,0,sizeof(int)*count, data_a,0,NULL,NULL);
    error = clEnqueueWriteBuffer(queue, input2, CL_TRUE,0,sizeof(int)*count, data_b,0,NULL,NULL);

    error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input1);
    error |= clSetKernelArg(kernel, 1 , sizeof(cl_mem), &input2);
    error |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &output);

    error = clEnqueueTask(queue, kernel,0,NULL,NULL);

    clFinish(queue);

    error = clEnqueueReadBuffer(queue, output,CL_TRUE,0,sizeof(int)*count, results,0,NULL,NULL);
    printf("out_matrix:\n");
    for(int i=0;i<m;i++)
    {
        for(int j=0;j<n;j++)
        {
            printf("%d ",results[i*n+j]);
        }
        printf("\n");
    }

}

void cleanup()
{
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseMemObject(input1);
    clReleaseMemObject(input2);
    clReleaseMemObject(output);
    clReleaseContext(context);
}