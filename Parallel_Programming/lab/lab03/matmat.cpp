#include <cstdlib>
#include <iostream>
#include <ctime>
#include <vector>
#include <chrono>

int main() {
  typedef float real;
  std::srand(std::time(0));
  std::cout << "Please give N: "; 
  int N;
  std::cin >> N;
  //Create 3 matrices
  std::vector<real> a(N*N);
  std::vector<real> b(N*N);
  std::vector<real> c(N*N);

  //Fill matrices with random values
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      a[i*N+j] = (real)std::rand()/(real)RAND_MAX;
      b[i*N+j] = (real)std::rand()/(real)RAND_MAX;
    }
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  #pragma omp parallel for
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        c[i*N+j] += b[i*N+k] * a[k*N+j];
      }
    }
  }
  //implement matrix-matrix multiplication here
  auto t2 = std::chrono::high_resolution_clock::now();

  //checksum
  real sum = 0;
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      sum += c[i*N+j];
  std::cout << sum << std::endl;
  std::cout << "took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
              << " milliseconds\n";
  std::cout << 2*N*N*N/std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()/1000.0/1000.0 << " GFLOPS/s" << std::endl;
  return 0;
}
