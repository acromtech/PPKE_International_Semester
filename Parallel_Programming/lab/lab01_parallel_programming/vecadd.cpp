#include <omp.h>
#include <iostream>
#include <chrono>
#include <vector>
#define N     100000000
int main (int argc, const char **argv)
{
  std::vector<float> a(N), b(N), c(N);
  /* Some initialisation */
  for (int i=0; i < N; i++)
    a[i] = b[i] = i * 1.0;

#pragma omp parallel for
  for (int i=0; i < N; i++)
    c[i] = a[i] + b[i];

  return 0;
}

