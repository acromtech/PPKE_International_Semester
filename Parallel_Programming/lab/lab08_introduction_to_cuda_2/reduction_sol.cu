#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <iostream>
#include <cuda.h>
#include <chrono>



//This is a little wrapper that checks for error codes returned by CUDA API calls
#define checkCuda(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

////////////////////////////////////////////////////////////////////////////////
// CPU routines
////////////////////////////////////////////////////////////////////////////////

double reduction_gold(double* idata, const unsigned int len) 
{
  double sum = 0;
  for(int i=0; i<len; i++) sum += idata[i];
  return sum;
}

////////////////////////////////////////////////////////////////////////////////
// GPU routines
////////////////////////////////////////////////////////////////////////////////

__global__ void reduction_atomic(double *g_odata, double *g_idata, int num_elements)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < num_elements)
    atomicAdd(g_odata, g_idata[tid]);
}

__global__ void reduction_shared1(double *g_odata, double *g_idata, int num_elements)
{
  __shared__ double sdata[512];
  int tid = threadIdx.x + blockIdx.x * blockDim.x;


  if (tid < num_elements)
    sdata[threadIdx.x] = g_idata[tid];
  else
    sdata[threadIdx.x] = 0;

  __syncthreads();

  if (threadIdx.x == 0) {
    double sum = 0;
    for (int i = 0; i < blockDim.x; i++)
      sum += sdata[i];
    atomicAdd(g_odata, sum);
  }
}

__global__ void reduction_shared2(double *g_odata, double *g_idata, int num_elements)
{
  __shared__ double sdata[512];
  int tid = threadIdx.x + blockIdx.x * blockDim.x;


  if (tid < num_elements)
    sdata[threadIdx.x] = g_idata[tid];
  else
    sdata[threadIdx.x] = 0;

  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    __syncthreads();
    if (threadIdx.x < s)
      sdata[threadIdx.x] += sdata[threadIdx.x + s];
  }

  if (threadIdx.x==0) {
    atomicAdd(g_odata, sdata[0]);
  }
}



////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

int main( int argc, const char** argv) 
{
  int num_elements;

  double *h_data, reference;
  double *d_idata, *d_odata;

  num_elements = 1<<25;

  // allocate host memory to store the input data
  // and initialize to integer values

  h_data = new double[num_elements];
      
  for(int i = 0; i < num_elements; i++) 
    h_data[i] = i;

  // compute reference solutions
  reference = reduction_gold(h_data, num_elements);

  // allocate device memory input and output arrays

  checkCuda( cudaMalloc((void**)&d_idata, (size_t)num_elements * sizeof(double)) );
  checkCuda( cudaMalloc((void**)&d_odata, sizeof(double)) );

  // copy host memory to device input array

  checkCuda( cudaMemcpy(d_idata, h_data, (size_t)num_elements * sizeof(double),
                              cudaMemcpyHostToDevice) );

  double zero = 0.0;
  checkCuda( cudaMemcpy(d_odata, &zero, sizeof(double),
                              cudaMemcpyHostToDevice) );

  auto t1 = std::chrono::high_resolution_clock::now();
  // execute the kernel
  int threads = 512;
  int blocks = (num_elements - 1) / threads + 1;
  reduction_atomic<<<blocks, threads>>>(d_odata, d_idata, num_elements);
  cudaDeviceSynchronize();
  auto t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
  std::cout << "Time: " << time_span.count() << " seconds." << std::endl;
  // copy result from device to host

  checkCuda( cudaMemcpy(h_data, d_odata, sizeof(double),
                              cudaMemcpyDeviceToHost) );

  // check results

  printf("reduction error = %f\n",h_data[0]-reference);

  checkCuda( cudaMemcpy(d_odata, &zero, sizeof(double),
                              cudaMemcpyHostToDevice) );
  t1 = std::chrono::high_resolution_clock::now();
  // execute the kernel
  reduction_shared1<<<blocks, threads>>>(d_odata, d_idata, num_elements);
  cudaDeviceSynchronize();
  t2 = std::chrono::high_resolution_clock::now();
  time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
  std::cout << "Time: " << time_span.count() << " seconds." << std::endl;
  // copy result from device to host

  checkCuda( cudaMemcpy(h_data, d_odata, sizeof(double),
                              cudaMemcpyDeviceToHost) );

  // check results

  printf("reduction error = %f\n",h_data[0]-reference);


  checkCuda( cudaMemcpy(d_odata, &zero, sizeof(double),
                              cudaMemcpyHostToDevice) );
  t1 = std::chrono::high_resolution_clock::now();
  // execute the kernel
  reduction_shared2<<<blocks, threads>>>(d_odata, d_idata, num_elements);
  cudaDeviceSynchronize();
  t2 = std::chrono::high_resolution_clock::now();
  time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
  std::cout << "Time: " << time_span.count() << " seconds." << std::endl;
  // copy result from device to host

  checkCuda( cudaMemcpy(h_data, d_odata, sizeof(double),
                              cudaMemcpyDeviceToHost) );

  // check results

  printf("reduction error = %f\n",h_data[0]-reference);

  // cleanup memory

  delete[] h_data;
  checkCuda( cudaFree(d_idata) );
  checkCuda( cudaFree(d_odata) );

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();
}

