#include <random>
#include <iostream>
#include <chrono>
#include <ctime>
#include <vector>
#include <cstdlib>
#include <omp.h>

int main() {
    int N,sum=0;
    std::cout << "Please give N: ";
    std::cin>>N;

    auto t1 = std::chrono::high_resolution_clock::now();
//Create random generator
    #pragma omp parallel reduction(+:sum)
    {
        std::default_random_engine generator(omp_get_thread_num());
    //Create distribution - double values between -0.5 and 0.5
        std::uniform_real_distribution<double> distribution(-0.5,0.5);
    //Get random value
        #pragma omp parallel for reduction(+:sum)
        for (int i = 0; i < N; i++) {
            double x = distribution(generator);
            double y = distribution(generator);
            if (x*x + y*y < 0.5*0.5)
                sum++;
        }
    }
    
// Your code here

    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Pi is approximately: " << 4.0*(double)sum/(double)N << std::endl;
    std::cout << "took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
              << " milliseconds\n";

    return 0;
}
