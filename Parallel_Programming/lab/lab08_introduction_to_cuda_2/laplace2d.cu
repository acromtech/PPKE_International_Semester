/* Copyright (c) 2012, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <chrono>

__device__ __forceinline__ double atomicMax(double *address, double val)
{
    unsigned long long ret = __double_as_longlong(*address);
    while(val > __longlong_as_double(ret))
    {
        unsigned long long old = ret;
        if((ret = atomicCAS((unsigned long long *)address, old, __double_as_longlong(val))) == old)
            break;
    }
    return __longlong_as_double(ret);
}

__global__ void stencil(int imax, int jmax, double *Anew, double *A, double *error) {
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int j = threadIdx.y + blockIdx.y*blockDim.y;
  extern __shared__ double local[];
  local[threadIdx.x + threadIdx.y*blockDim.x] = 0.0;

  if (i < imax+1 && j < jmax+1 && i>0 && j>0) {
    Anew[(j)*(imax+2)+i] = 0.25f * ( A[(j)*(imax+2)+i+1] + A[(j)*(imax+2)+i-1]
        + A[(j-1)*(imax+2)+i] + A[(j+1)*(imax+2)+i]);
    local[threadIdx.x + threadIdx.y*blockDim.x] = fabs(Anew[(j)*(imax+2)+i]-A[(j)*(imax+2)+i]);
  }
  int threads = blockDim.x * blockDim.y;
  for (int s = threads>>1; s > 0; s >>= 1) {
    __syncthreads();
    if (threadIdx.x + threadIdx.y*blockDim.x < s)
        local[threadIdx.x + threadIdx.y*blockDim.x] = fmax(local[threadIdx.x + threadIdx.y*blockDim.x], local[threadIdx.x + threadIdx.y*blockDim.x + s]);
  }
  if (threadIdx.x ==0 && threadIdx.y ==0) atomicMax(error, local[0]);
  // error = fmax( error, fabs(Anew[(j)*(imax+2)+i]-A[(j)*(imax+2)+i]));
}

__global__ void copy(int imax, int jmax, double *Anew, double *A) {

  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int j = threadIdx.y + blockIdx.y*blockDim.y;
  if (i < imax+1 && j < jmax+1 && i>0 && j>0)
    A[(j)*(imax+2)+i] = Anew[(j)*(imax+2)+i];
}

int main(int argc, char** argv)
{
    //Size along y
    int jmax = 4094;
    //Size along x
    int imax = 4094;
    int iter_max = 1000;

    const double pi  = 2.0 * asin(1.0);
    const double tol = 1.0e-5;
    double error     = 1.0;

    double *A;
    double *Anew;
    double *y0;

    A    = (double *)malloc((imax+2) * (jmax+2) * sizeof(double));
    Anew = (double *)malloc((imax+2) * (jmax+2) * sizeof(double));
    y0   = (double *)malloc((imax+2) * sizeof(double));

    memset(A, 0, (imax+2) * (jmax+2) * sizeof(double));

    // set boundary conditions
    for (int i = 0; i < imax+2; i++)
      A[(0)*(imax+2)+i]   = 0.0;

    for (int i = 0; i < imax+2; i++)
      A[(jmax+1)*(imax+2)+i] = 0.0;

    for (int j = 0; j < jmax+2; j++)
    {
        y0[j] = sin(pi * j / (jmax+1));
        A[(j)*(imax+2)+0] = y0[j];
    }

    for (int j = 0; j < imax+2; j++)
    {
        y0[j] = sin(pi * j/ (jmax+1));
        A[(j)*(imax+2)+imax+1] = y0[j]*exp(-pi);
    }

    printf("Jacobi relaxation Calculation: %d x %d mesh\n", imax+2, jmax+2);

    auto t1 = std::chrono::high_resolution_clock::now();
    int iter = 0;

    for (int i = 1; i < imax+2; i++)
       Anew[(0)*(imax+2)+i]   = 0.0;

    for (int i = 1; i < imax+2; i++)
       Anew[(jmax+1)*(imax+2)+i] = 0.0;

    for (int j = 1; j < jmax+2; j++)
        Anew[(j)*(imax+2)+0]   = y0[j];

    for (int j = 1; j < jmax+2; j++)
        Anew[(j)*(imax+2)+jmax+1] = y0[j]*expf(-pi);


    double *d_A, *d_Anew, *d_error;
    size_t size = (imax+2)*(jmax+2)*sizeof(double);
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_Anew, size);
    cudaMalloc(&d_error, sizeof(double));
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Anew, Anew, size, cudaMemcpyHostToDevice);
    while ( error > tol && iter < iter_max )
    {
        error = 0.0;
        cudaMemcpy(d_error, &error, sizeof(double), cudaMemcpyHostToDevice);
        dim3 threads(16,4);
        dim3 blocks((imax+2)/16, (jmax+2)/4);
        stencil<<<blocks, threads>>>(imax, jmax, d_Anew, d_A, d_error);
        cudaMemcpy(&error, d_error, sizeof(double), cudaMemcpyDeviceToHost);

        copy<<<blocks, threads>>>(imax, jmax, d_Anew, d_A);
        if(iter % 100 == 0) printf("%5d, %0.6f\n", iter, error);

        iter++;
    }
    cudaMemcpy(A, d_A, size, cudaMemcpyDeviceToHost);

    auto t2 = std::chrono::high_resolution_clock::now();
    double runtime = std::chrono::duration<double>(t2 - t1).count();

    printf(" total: %f s\n", runtime);
}
