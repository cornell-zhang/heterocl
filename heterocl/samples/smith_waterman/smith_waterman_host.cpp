#include <iostream>
using namespace std;

#define lenA 10
#define lenB 10

void default_function(int ind, int* seqA, int* seqB, int* consensusA,
    int* consensusB, int* maxtrix);

int main(int argc, char **argv) {
  int _seqA[lenA] = {1,2,3,4,3,4,1,3,2,3};
  int _seqB[lenB] = {3,2,2,4,3,3,4,3,3,2};
  int _outA[lenA + lenB + 2] = {0};
  int _outB[lenA + lenB + 2] = {0};
  int _matrix[(lenA + 1) * (lenB + 1)];

  // Reference result
  int _refA[] = {3,4,3,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  int _refB[] = {3,4,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  int _refMatrix[] = {
    0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,
    0,0,1,1,0,0,0,0,0,0,0,
    0,1,0,0,0,1,1,0,1,1,0,
    0,0,0,0,1,0,0,2,0,0,0,
    0,1,0,0,0,2,1,0,3,1,0,
    0,0,0,0,1,0,0,2,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,
    0,1,0,0,0,1,1,0,1,1,0,
    0,0,2,1,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0};

  default_function(0, _seqA, _seqB, _outA, _outB, _matrix);

  // Check result
  bool pass = true;
  for (int i = 0; i < lenA + lenB + 2; ++i) {
    if (_outA[i] != _refA[i]) {
      cout << "Expect outA[" << i << "] = " << _refA[i] << " but ";
      cout << _outA[i] << endl;
      pass = false;
    }
  }
  for (int i = 0; i < lenA + lenB + 2; ++i) {
    if (_outB[i] != _refB[i]) {
      cout << "Expect outB[" << i << "] = " << _refB[i] << " but ";
      cout << _outB[i] << endl;
      pass = false;
    }
  }
  for (int i = 0; i < lenA + 1; ++i) {
    for (int j = 0; j < lenB + 1; ++j) {
      if (_matrix[i * (lenB + 1) + j] != _refMatrix[i * (lenB + 1) + j]) {
        cout << "Expect matrix[" << i << "][ " << j << "]= ";
        cout << _refMatrix[i * (lenB + 1) + j] << " but ";
        cout << _matrix[i * (lenB + 1) + j] << endl;
        pass = false;
      }
    }
  }

  if (!pass) {
    cout << "Result checking failed." << endl;
    return 1;
  }
  return 0;
}
