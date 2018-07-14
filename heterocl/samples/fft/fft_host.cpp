#include <iostream>
#include <fstream>
using namespace std;

#define L 1024

void default_function(float* placeholder0, float* placeholder1,
    int* placeholder2, float* F_real, float* F_imag);

int main(int argc, char **argv) {
  float x_real[L], out_real[L];
  float x_imag[L], out_imag[L];
  int table[L];

  // Read input
  ifstream f;
  f.open("input.dat", ios::in);
  for (int i = 0; i < L; ++i)
    f >> x_real[i];
  for (int i = 0; i < L; ++i)
    f >> x_imag[i];
  for (int i = 0; i < L; ++i)
    f >> table[i];
  f.close();

  // Compute
  default_function(x_real, x_imag, table, out_real, out_imag);

  // Output
  ofstream of;
  of.open("output.dat", ios::out);
  of << out_real[0];
  for (int i = 1; i < L; ++i)
    of << "\t" << out_real[i];
  of << endl;
  of << out_imag[0];
   for (int i = 1; i < L; ++i)
    of << "\t" << out_imag[i];
  of << endl;

  return 0;
}
