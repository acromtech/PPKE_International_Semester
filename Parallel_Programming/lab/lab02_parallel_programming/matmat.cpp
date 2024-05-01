#include <random>
#include <iostream>
#include <vector>
#include <chrono>

int main(int argc, char **argv) {
  int N;
  if (argc > 1)
    N = atoi(argv[1]);
  else {
    std::cout << "Please give N: ";
    std::cin>>N;
  }
  //Create 3 matrices
  std::vector<float> a(N*N);
  std::vector<float> b(N*N);
  std::vector<float> c(N*N);

  //Create random generator with 0 seed
  std::default_random_engine generator(0);
  //Create distribution - float values between 0.0 and 1.0
  std::uniform_real_distribution<float> distribution(0.0,1.0);
  //Fill matrices with random values
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      a[i*N+j] = distribution(generator);
      b[i*N+j] = distribution(generator);
    }
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  //implement matrix-matrix multiplication here
  #pragma omp parallel for
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        c[i*N+j] += a[i*N+k] * b[k*N+j];
      }
    }
  }
  auto t2 = std::chrono::high_resolution_clock::now();

  //checksum
  float sum = 0;
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      sum += c[i*N+j];
  std::cout << sum << std::endl;
  std::cout << "took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
              << " milliseconds\n";
  std::cout << 2*N*N*N/std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()/1000.0/1000.0 << " GFLOPS/s" << std::endl;
  
   // Calculate theoretical number of operations
  long long operations = 2LL * N * N * N;

  // Calculate operations per second
  double duration_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() / 1000.0;
  double operations_per_second = operations / duration_seconds;

  std::cout << "Checksum: " << sum << std::endl;
  std::cout << "Time taken: " << duration_seconds << " seconds\n";
  std::cout << "Theoretical number of operations: " << operations << std::endl;
  std::cout << "Operations per second: " << operations_per_second << " ops/s" << std::endl;
  
  return 0;
}
