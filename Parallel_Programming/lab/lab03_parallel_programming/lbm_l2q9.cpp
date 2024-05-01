#include <iostream>
#include <fstream>
#include <cstdlib>
#include <random>
#include <chrono>
#include <omp.h>

static inline int mod(int v, int m) {
    int val = v % m;
    if (val < 0) val = m + val;
    return val;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Please provide two parameters for the mesh size\n");
        exit(1);
    }
    int NX = atoi(argv[1]);
    int NY = atoi(argv[2]);
    const double OMEGA = 1.0;
    const double rho0 = 1.0;
    const double deltaUX = 10e-6;

    const double W[] = {4.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 9.0, 1.0 / 36.0};
    const int cx[] = {0, 0, 1, 1, 1, 0, -1, -1, -1};
    const int cy[] = {0, 1, 1, 0, -1, -1, -1, 0, 1};

    const int opposite[] = {0, 5, 6, 7, 8, 1, 2, 3, 4};

    //Allocate memory
    int* __restrict__ SOLID = new int[NX * NY];
    //Work arrays
    double* __restrict__ workArray = new double[NX * NY * 9];
    double* __restrict__ N_SOLID = new double[NX * NY * 9];
    double* __restrict__ rho = new double[NX * NY];
    double* __restrict__ ux = new double[NX * NY];
    double* __restrict__ uy = new double[NX * NY];

    //Generate random obstacles
    std::default_random_engine generator(0);
    //Create distribution - double values between 0.0 and 1.0
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    //Parallelize with care - may not get the same answers
#pragma omp parallel for collapse(2)
    for (int j = 0; j < NY; j++) {
        for (int i = 0; i < NX; i++) {
            if (distribution(generator) >= 0.7)
                SOLID[j * NX + i] = 1;
            else
                SOLID[j * NX + i] = 0;
        }
    }

    //Initial values
    double* __restrict__ N = new double[NX * NY * 9];
#pragma omp parallel for collapse(3)
    for (int j = 0; j < NY; j++) {
        for (int i = 0; i < NX; i++) {
            for (int f = 0; f < 9; f++) {
                N[(j * NX + i) * 9 + f] = rho0 * W[f];
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    //Main time loop
    for (int t = 0; t < 1000; t++) {

        // Backup values
        auto backup_start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for collapse(3)
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                for (int f = 0; f < 9; f++) {
                    workArray[(j * NX + i) * 9 + f] = N[(j * NX + i) * 9 + f];
                }
            }
        }
        auto backup_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> backup_time = backup_end - backup_start;
        int bytes_per_iter = (NX * NY * 9) * (8 + 8);
        double bw = ((bytes_per_iter * 1000.0) / (backup_time.count() * 1024 * 1024 * 1024));
        std::cout << "kernel 1:\t" << bytes_per_iter <<"\t" <<backup_time.count() << "s \t"<<bw <<"\n";

        // Gather neighbour values
        auto gather_start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for collapse(3)
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                for (int f = 1; f < 9; f++) {
                    N[(j * NX + i) * 9 + f] = workArray[(mod(j - cy[f], NY) * NX + mod(i - cx[f], NX)) * 9 + f];
                }
            }
        }
        auto gather_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> gather_time = gather_end - gather_start;
        bytes_per_iter = (NX * NY * 9) * (8 + 8);
        bw = ((bytes_per_iter * 1000.0) / (gather_time.count() * 1024 * 1024 * 1024));
        std::cout << "kernel 2:\t" << bytes_per_iter <<"\t" <<gather_time.count() << "s \t"<<bw <<"\n";        

        // Bounce back from solids, no collision
        auto bounce_start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for collapse(2)
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                if (SOLID[j * NX + i] == 1) {
                    for (int f = 0; f < 9; f++) {
                        N_SOLID[(j * NX + i) * 9 + opposite[f]] = N[(j * NX + i) * 9 + f];
                    }
                }
            }
        }
        auto bounce_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> bounce_time = bounce_end - bounce_start;
        bytes_per_iter = (NX * NY * 9) * (8 + 8);
        bw = ((bytes_per_iter * 1000.0) / (bounce_time.count() * 1024 * 1024 * 1024));
        std::cout << "kernel 3:\t" << bytes_per_iter <<"\t" <<bounce_time.count() << "s \t"<<bw <<"\n";        
   
        // Calculate rho
        auto rho_start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for collapse(2)
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                rho[j * NX + i] = 0;
                for (int f = 0; f < 9; f++) {
                    rho[j * NX + i] += N[(j * NX + i) * 9 + f];
                }
            }
        }
        auto rho_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> rho_time = rho_end - rho_start;
        bytes_per_iter = (NX * NY * 9) * 8 + (NX * NY) * 8;
        bw = ((bytes_per_iter * 1000.0) / (rho_time.count() * 1024 * 1024 * 1024));
        std::cout << "kernel 4:\t" << bytes_per_iter <<"\t" <<rho_time.count() << "s \t"<< bw << "\n";   

        // Calculate ux
        auto ux_start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for collapse(2)
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                ux[j * NX + i] = 0;
                for (int f = 0; f < 9; f++) {
                    ux[j * NX + i] += N[(j * NX + i) * 9 + f] * cx[f];
                }
                ux[j * NX + i] = ux[j * NX + i] / rho[j * NX + i] + deltaUX;
            }
        }
        auto ux_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> ux_time = ux_end - ux_start;
        bytes_per_iter = (NX * NY) * (8 + 8 + 8) + 8;
        bw = ((bytes_per_iter * 1000.0) / (ux_time.count() * 1024 * 1024 * 1024));
        std::cout << "kernel 5:\t" << bytes_per_iter <<"\t" <<ux_time.count() << "s \t"<< bw << "\n";  

        // Calculate uy
        auto uy_start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for collapse(2)
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                uy[j * NX + i] = 0;
                for (int f = 0; f < 9; f++) {
                    uy[j * NX + i] += N[(j * NX + i) * 9 + f] * cy[f];
                }
                uy[j * NX + i] = uy[j * NX + i] / rho[j * NX + i];
            }
        }
        auto uy_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> uy_time = uy_end - uy_start;
        bytes_per_iter = (NX * NY) * (8 + 8 + 8) ;
        bw = ((bytes_per_iter * 1000.0) / (uy_time.count() * 1024 * 1024 * 1024));
        std::cout << "kernel 6:\t" << bytes_per_iter <<"\t" <<uy_time.count() << "s \t"<< bw << "\n";  

        // Calculate workArray
        auto workArray_start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for collapse(3)
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                for (int f = 0; f < 9; f++) {
                    workArray[(j * NX + i) * 9 + f] = ux[j * NX + i] * cx[f] + uy[j * NX + i] * cy[f];
                }
            }
        }
        auto workArray_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> workArray_time = workArray_end - workArray_start;
        bytes_per_iter = (NX * NY *9)* 8 +  (NX * NY) *(8+8) + 9*(4+4);
        bw = ((bytes_per_iter * 1000.0) / (workArray_time.count() * 1024 * 1024 * 1024));
        std::cout << "kernel 7:\t" << bytes_per_iter <<"\t" <<workArray_time.count() << "s \t"<< bw << "\n";  

        // Perform first collision step
        auto collision1_start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for collapse(3)
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                for (int f = 0; f < 9; f++) {
                    workArray[(j * NX + i) * 9 + f] = (3 + 4.5 * workArray[(j * NX + i) * 9 + f]) * workArray[(j * NX + i) * 9 + f];
                }
            }
        }
        auto collision1_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> collision1_time = collision1_end - collision1_start;
        bytes_per_iter = (NX * NY * 9) * (8 + 8 + 8);
        bw = ((bytes_per_iter * 1000.0) / (collision1_time.count() * 1024 * 1024 * 1024));
        std::cout << "kernel 8:\t" << bytes_per_iter <<"\t" <<collision1_time.count() << "s \t"<< bw <<"\n";  

        // Perform second collision step
        auto collision2_start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for collapse(3)
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                for (int f = 0; f < 9; f++) {
                    workArray[(j * NX + i) * 9 + f] -= 1.5 * (ux[j * NX + i] * ux[j * NX + i] + uy[j * NX + i] * uy[j * NX + i]);
                }
            }
        }
        auto collision2_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> collision2_time = collision2_end - collision2_start;
        bytes_per_iter = (NX * NY * 9) * 8 +(NX * NY) *(8 + 8 + 8 + 8);
        bw = ((bytes_per_iter * 1000.0) / (collision2_time.count() *1024*1024*1024));
        std::cout << "kernel 9:\t" << bytes_per_iter <<"\t" <<collision2_time.count() << "s \t"<< bw << "\n";  

        // Calculate new distribution functions
        auto distribution_start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for collapse(3)
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                for (int f = 0; f < 9; f++) {
                    N[(j * NX + i) * 9 + f] = (1 + workArray[(j * NX + i) * 9 + f]) * W[f] * rho[j * NX + i];
                }
            }
        }
        auto distribution_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> distribution_time = distribution_end - distribution_start;
        bytes_per_iter = (NX * NY * 9) * (8 + 8) + 9 * 8 + (NX * NY) * 8 + 4;
        bw = ((bytes_per_iter * 1000.0) / (distribution_time.count() * 1024 * 1024 * 1024));
        std::cout << "kernel 10:\t" << bytes_per_iter << "\t" << distribution_time.count() << "s \t" << bw << "\n";

        // Restore solid cells
        auto restore_start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for collapse(2)
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                if (SOLID[j * NX + i] == 1) {
                    for (int f = 0; f < 9; f++) {
                        N[(j * NX + i) * 9 + f] = N_SOLID[(j * NX + i) * 9 + f];
                    }
                }
            }
        }
        auto restore_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> restore_time = restore_end - restore_start;
        bytes_per_iter = (NX * NY * 9) * (8 + 8) ;
        bw = ((bytes_per_iter * 1000.0) / (restore_time.count() * 1024 * 1024 * 1024));
        std::cout << "kernel 11:\t" << bytes_per_iter << "\t" << restore_time.count() << "s \t" << bw << "\n";

        // Calculate kinetic energy
        auto kinetic_start = std::chrono::high_resolution_clock::now();
        double energy = 0;
#pragma omp parallel for reduction(+ \
                                    : energy)
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                energy += ux[j * NX + i] * ux[j * NX + i] + uy[j * NX + i] * uy[j * NX + i];
            }
        }
        if (t % 100 == 0) std::cout << energy << std::endl;
        auto kinetic_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> kinetic_time = kinetic_end - kinetic_start;
        bytes_per_iter = 8 + (NX * NY ) *(8+8+8+8) ;
        bw = ((bytes_per_iter * 1000.0) / (kinetic_time.count() * 1024 * 1024 * 1024));
        std::cout << "kernel 12:\t" << bytes_per_iter << "\t" <<kinetic_time.count() << "s \t" << bw << "\n";

    }

    if (true) {
        std::ofstream myfile;
        myfile.open("output_velocity.csv");
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                myfile << SOLID[j * NX + i] << ", " << ux[j * NX + i] << ", " << uy[j * NX + i] << std::endl;
            }
        }
        myfile.close();
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    double elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() / 1000.0;
    std::cout << "Elapsed time: " << elapsed_seconds << " seconds" << std::endl;

}
