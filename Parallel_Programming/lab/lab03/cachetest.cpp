// Cache test app
//Source: http://www.roguewave.com/getattachment/b6524fa0-2f6f-4498-9875-194886ca8def/CPU-Cache-Optimization?sitename=RogueWave
#include <iostream>
#include <omp.h>
#include <stdlib.h>

struct data {
  int a; int b; 
  int c; int d;
};
typedef struct data DATA;
int main() {
  DATA * pMyData = (DATA*)malloc(10*1024*1024*sizeof(DATA));
  double t1 = omp_get_wtime();
  for (long i=0; i<10*1024*1024; i++)
  {
    pMyData[i].a = pMyData[i].b;
  }
  double t2 = omp_get_wtime();
  std::cout << "Total time " << t2-t1 << std::endl;
  delete[] pMyData;
  return 0;
}
