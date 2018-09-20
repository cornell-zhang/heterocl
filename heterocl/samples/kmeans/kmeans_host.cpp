#include <stdlib.h>
#include <limits.h>
#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>
using namespace std;

#define K    16
#define N    320
#define D    32

#ifdef MCC_ACC
#include MCC_ACC_H_FILE
#else
void default_function(int* X0, int* centers0);
#endif

void clustering(int *X0, int *centers) {
  int count[K] = {0};

  for (int i = 0; i < N; ++i) {
    int idx = 0;
    int minDis = INT_MAX;
    for (int i1 = 0; i1 < K; ++i1) {
      int dis = 0;
      for (int i2 = 0; i2 < D; ++i2) {
        dis += (((long) X0[i2 + i * D]) - ((long) centers[i2 + i1 * D])) *
               (((long) X0[i2 + i * D]) - ((long) centers[i2 + i1 * D]));
      }
      if (dis < minDis) {
        minDis = dis;
        idx = i1;
      }
    }
    count[idx]++;
  }

  for (int z = 0; z < K; ++z)
    cout << "Cluster " << z << ": " << count[z] << endl;

  return ;
}

int main(int argc, char **argv) {
  int X0[N * D];
  int centers0[K * D];

  srand(0);

#ifdef MCC_ACC
  __merlin_init(argv[argc-1]);
#endif

  // Prepare data
  for (int i = 0; i < N * D; ++i)
    X0[i] = rand() % 100;

  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < D; ++j)
      centers0[i * D + j] = X0[i * D + j];
  }

  // Compute
#ifdef MCC_ACC
  __merlin_default_function(X0, centers0);
#else
  default_function(X0, centers0);
#endif

  // Evaluate
  clustering(X0, centers0);

#ifdef MCC_ACC
  __merlin_release();
#endif

  return 0;
}
