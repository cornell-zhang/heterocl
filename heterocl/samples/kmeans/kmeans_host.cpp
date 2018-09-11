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

void default_function(int* X0, int* centers0);

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
  int *X0;
  int *centers0;

  srand(0);

  // Prepare data
  X0 = (int *)malloc(sizeof(int) * N * D);
  for (int i = 0; i < N * D; ++i)
    X0[i] = rand() % 100;

  centers0 = (int *)malloc(sizeof(int) * K * D);
  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < D; ++j)
      centers0[i * D + j] = X0[i * D + j];
  }

  // Compute
  default_function(X0, centers0);

  // Evaluate
  clustering(X0, centers0);

  return 0;
}
