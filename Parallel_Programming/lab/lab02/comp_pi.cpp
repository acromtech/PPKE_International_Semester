#include <random>
#include <iostream>
#include <chrono>
#include <ctime>
#include <vector>
#include <cstdlib>
#include <omp.h>

int main(int argc, char **argv) {
    int N,sum=0;
    if (argc > 1)
      N = atoi(argv[1]);
    else {
      std::cout << "Please give N: ";
      std::cin>>N;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    //Create random generator with 0 seed
    std::default_random_engine generator(0);
    //Create distribution - double values between -0.5 and 0.5
    std::uniform_real_distribution<double> distribution(-0.5,0.5);
    //Get random value
    double x = distribution(generator);
    double y = distribution(generator);

    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Pi is approximately: " << std::endl;
    std::cout << "took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
              << " milliseconds\n";

    return 0;
}
