#include <iostream>
#include <stdlib.h>
#include <omp.h>
#include <chrono>

#define N 10000000
void saxpy(int n, float alpha, float * X, float * Y) {
  for (int i=0; i<n; i++) 
    Y[i] = alpha*X[i] + Y[i]; 
}

int main() {
  float *x = new float[N];
  float *y = new float[N];
  float alpha = 2.0f;
  auto t1 = std::chrono::high_resolution_clock::now();
  saxpy(N,alpha,x,y);
  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout << "took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
              << " milliseconds\n";
  return 0;
}

