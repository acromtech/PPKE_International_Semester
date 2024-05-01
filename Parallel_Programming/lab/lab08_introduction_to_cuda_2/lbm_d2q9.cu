#include <iostream>
#include <fstream>
#include <cstdlib>
#include <random>
#include <chrono>
#include <cuda.h>
#include <curand_kernel.h>

// CUDA Kernel to initialize the SOLID array with random values
__global__ void initializeSolid(int *SOLID, int NX, int NY) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NX * NY) {
        curandState state;
        curand_init(clock64(), idx, 0, &state); // Initialize the random number generator
        SOLID[idx] = curand_uniform(&state) >= 0.7 ? 1 : 0; // Generate random number and set SOLID accordingly
    }
}

// CUDA Kernel to perform the LBM computation
__global__ void performLBM(double *N, double *workArray, int *SOLID, double *rho, double *ux, double *uy, 
                           double *N_SOLID, const double *W, const int *cx, const int *cy, const int *opposite,
                           double deltaUX, int NX, int NY) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NX * NY * 9) {
        // Backup values
        workArray[idx] = N[idx];
        
        // Gather neighbour values
        int i = idx % NX;
        int j = (idx / NX) % NY;
        int f = idx / (NX * NY);
        if (f > 0) {
            N[idx] = workArray[((((j - cy[f]) % NY + NY) % NY) * NX + ((i - cx[f]) % NX + NX) % NX) * 9 + f];
        }
        
        // Bounce back from solids, no collision
        int gridIdx = idx / 9;
        i = gridIdx % NX;
        j = gridIdx / NX;
        if (SOLID[gridIdx] == 1) {
            N_SOLID[idx] = N[idx];
        }

        // Calculate rho
        if (f == 0) {
            double localRho = 0;
            for (int k = 0; k < 9; k++) {
                localRho += N[(j * NX + i) * 9 + k];
            }
            rho[gridIdx] = localRho;
        }

        // Calculate ux
        if (f == 0) {
            double localUx = 0;
            for (int k = 0; k < 9; k++) {
                localUx += N[(j * NX + i) * 9 + k] * cx[k];
            }
            ux[gridIdx] = (localUx / rho[gridIdx]) + deltaUX;
        }

        // Calculate uy
        if (f == 0) {
            double localUy = 0;
            for (int k = 0; k < 9; k++) {
                localUy += N[(j * NX + i) * 9 + k] * cy[k];
            }
            uy[gridIdx] = localUy / rho[gridIdx];
        }

        // Calculate workArray
        if (f == 0) {
            double localUx = ux[gridIdx];
            double localUy = uy[gridIdx];
            for (int k = 0; k < 9; k++) {
                workArray[(j * NX + i) * 9 + k] = localUx * cx[k] + localUy * cy[k];
            }
        }

        // Perform first collision step
        if (f == 0) {
            for (int k = 0; k < 9; k++) {
                workArray[(j * NX + i) * 9 + k] = (3 + 4.5 * workArray[(j * NX + i) * 9 + k]) * workArray[(j * NX + i) * 9 + k];
            }
        }

        // Perform second collision step
        if (f == 0) {
            double localUx = ux[gridIdx];
            double localUy = uy[gridIdx];
            for (int k = 0; k < 9; k++) {
                workArray[(j * NX + i) * 9 + k] -= 1.5 * (localUx * localUx + localUy * localUy);
            }
        }

        // Calculate new distribution functions
        if (f == 0) {
            double localRho = rho[gridIdx];
            for (int k = 0; k < 9; k++) {
                N[(j * NX + i) * 9 + k] = (1 + workArray[(j * NX + i) * 9 + k]) * W[k] * localRho;
            }
        }

        // Restore solid cells
        if (SOLID[gridIdx] == 1) {
            N[idx] = N_SOLID[idx];
        }
    }
}

int main(int argc, char** argv) {
    int NX = atoi(argv[1]);
    int NY = atoi(argv[2]);
    const double deltaUX = 10e-6;

    const double W[] = {4.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 9.0, 1.0 / 36.0};
    const int cx[] = {0, 0, 1, 1, 1, 0, -1, -1, -1};
    const int cy[] = {0, 1, 1, 0, -1, -1, -1, 0, 1};
    const int opposite[] = {0, 5, 6, 7, 8, 1, 2, 3, 4};

    // Allocate memory on host
    int* SOLID = new int[NX * NY];
    double* N = new double[NX * NY * 9];
    double* workArray = new double[NX * NY * 9];
    double* N_SOLID = new double[NX * NY * 9];
    double* rho = new double[NX * NY];
    double* ux = new double[NX * NY];
    double* uy = new double[NX * NY];

    // Allocate memory on device
    int *d_SOLID;
    double *d_N, *d_workArray, *d_N_SOLID, *d_rho, *d_ux, *d_uy;
    cudaMalloc((void **)&d_SOLID, NX * NY * sizeof(int));
    cudaMalloc((void **)&d_N, NX * NY * 9 * sizeof(double));
    cudaMalloc((void **)&d_workArray, NX * NY * 9 * sizeof(double));
    cudaMalloc((void **)&d_N_SOLID, NX * NY * 9 * sizeof(double));
    cudaMalloc((void **)&d_rho, NX * NY * sizeof(double));
    cudaMalloc((void **)&d_ux, NX * NY * sizeof(double));
    cudaMalloc((void **)&d_uy, NX * NY * sizeof(double));

    // Initialize random seed
    srand(0);

    // Initialize solid array on device
    initializeSolid<<<(NX * NY + 255) / 256, 256>>>(d_SOLID, NX, NY);

    auto t1 = std::chrono::high_resolution_clock::now();

    // Main time loop
    for (int t = 0; t < 1000; t++) {
        // Call CUDA kernel to perform LBM computation
        performLBM<<<(NX * NY * 9 + 255) / 256, 256>>>(d_N, d_workArray, d_SOLID, d_rho, d_ux, d_uy, d_N_SOLID,
                                                        W, cx, cy, opposite, deltaUX, NX, NY);
        cudaDeviceSynchronize(); // Wait for all kernels to finish
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    double elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() / 1000.0;
    std::cout << "Elapsed time: " << elapsed_seconds << " seconds" << std::endl;

    // Copy results back to host
    cudaMemcpy(SOLID, d_SOLID, NX * NY * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(N, d_N, NX * NY * 9 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(rho, d_rho, NX * NY * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(ux, d_ux, NX * NY * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(uy, d_uy, NX * NY * sizeof(double), cudaMemcpyDeviceToHost);

    // Free memory on device
    cudaFree(d_SOLID);
    cudaFree(d_N);
    cudaFree(d_workArray);
    cudaFree(d_N_SOLID);
    cudaFree(d_rho);
    cudaFree(d_ux);
    cudaFree(d_uy);

    // Free memory on host
    delete[] SOLID;
    delete[] N;
    delete[] workArray;
    delete[] N_SOLID;
    delete[] rho;
    delete[] ux;
    delete[] uy;

    return 0;
}
