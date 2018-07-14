#include <stdlib.h>
#include <limits.h>
#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>
using namespace std;

#ifdef NO_FPGA
int popcount(unsigned long num) {
  int out = 0;
  for (int i = 0; i < 49; ++i)
    out += ((num & (1L << i)) >> i);
  return out;
}

void update_knn(unsigned long *dist, unsigned char *knn_mat, int x, int y) {
  int id = 0;
  for (int i = 0; i < 3; ++i) {
    if (knn_mat[x * 3 + i] > knn_mat[x * 3 + id])
      id = i;
  }
  if (dist[x * 1800 + y] < knn_mat[x * 3 + id])
    knn_mat[x * 3 + id] = dist[x * 1800 + y];
  return ;
}

void default_function(unsigned long test_image, unsigned long *train_image,
    unsigned char *knn_mat) {

  unsigned long diff[18000];
  unsigned long dist[18000];

  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 3; ++j)
      knn_mat[i * 3 + j] = 50;

    for (int j = 0; j < 1800; ++j) {
      diff[i * 1800 + j] = train_image[i * 1800 + j] ^ test_image;
    }
    for (int j = 0; j < 1800; ++j) {
      dist[i * 1800 + j] = popcount(diff[i * 1800 + j]);
    }
    for (int j = 0; j < 1800; ++j) {
      update_knn(dist, knn_mat, i, j);
    }
  }
  return ;
}
#else
void default_function(unsigned long test_image, unsigned long* train_images,
    unsigned char* knn_mat);
#endif

void read_train_file(string filename, unsigned long *train_image) {
  ifstream f;
  f.open(filename.c_str(), ios::in);
  if (!f.is_open()) {
    cout << "Open " << filename << " failed" << endl;
    exit(1);
  }

  int cnt = 0;
  while (!f.eof()) {
    string str;
    f >> str;
    unsigned long val = strtoul(str.substr(0, str.length() - 1).c_str(),
        NULL, 0);
    train_image[cnt++] = val;
  }
  f.close();
  return ;
}

void read_test_file(string filename, unsigned long *test_image,
  int *test_label) {

  ifstream f;
  f.open(filename.c_str(), ios::in);
  if (!f.is_open()) {
    cout << "Open " << filename << " failed" << endl;
    exit(1);
  }
  int cnt = 0;
  while (!f.eof()) {
    string str;
    f >> str;
    unsigned long val = strtoul(str.substr(0, str.length() - 2).c_str(),
        NULL, 0);
    test_image[cnt] = val;
    int label = str[str.length() - 1] - '0';
    test_label[cnt++] = label;
  }
  f.close();
  return ;
}

int vote(unsigned char *knn_mat) {
  int score[10] = {0};

  for (int i = 0; i < 30; i += 3)
    sort(knn_mat + i, knn_mat + i + 3);

  for (int i = 0; i < 3; ++i) {
    int m = INT_MAX;
    int id = 0;
    for (int j = 0; j < 10; ++j) {
      if (knn_mat[j * 3 + i] < m) {
        m = knn_mat[j * 3 + i];
        id = j;
      }
    }
    score[id]++;
  }
 
  int vid = 0;
  int vm = -1;
  for (int i = 0; i < 10; ++i) {
    if (score[i] > vm) {
      vm = score[i];
      vid = i;
    }
  }
  return vid;
}

int main(int argc, char **argv) {
  unsigned long *train_image;
  unsigned long *test_image;
  int *test_label;

  // Prepare data
  train_image = (unsigned long *)malloc(sizeof(unsigned long) * 10 * 1800);
  for (int i = 0; i < 10; ++i) {
    read_train_file("./data/training_set_" + to_string(i) + ".dat",
        &train_image[i * 1800]);
  }

  test_image = (unsigned long *)malloc(sizeof(unsigned long) * 180);
  test_label = (int *)malloc(sizeof(int) * 180);
  read_test_file("./data/testing_set.dat", test_image, test_label);

  // Compute
  int correct = 0;
  for (int i = 0; i < 180; ++i) {
    unsigned char *knn_mat = (unsigned char *)malloc(
        sizeof(unsigned char) * 3 * 10);
    default_function(test_image[i], train_image, knn_mat);
    if (vote(knn_mat) == test_label[i])
      correct++;
  }
  cout << (float) correct / 180 << endl;


  return 0;
}
