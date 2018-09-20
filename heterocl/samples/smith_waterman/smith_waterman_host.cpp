#include <iostream>
#include <string.h>
using namespace std;

#define lenA 128
#define lenB 128
#define num 1024

#ifdef MCC_ACC
#include MCC_ACC_H_FILE
#else
void default_function(unsigned char* seqA, unsigned char* seqB,
                      unsigned char* consensusA, unsigned char* consensusB);
#endif

int main(int argc, char **argv) {
  unsigned char *_seqA;
  unsigned char *_seqB;
  unsigned char *_outA;
  unsigned char *_outB;

#ifdef MCC_ACC
  __merlin_init(argv[argc-1]);
#endif

  _seqA = (unsigned char *)malloc(num * lenA * sizeof(unsigned char));
  _seqB = (unsigned char *)malloc(num * lenB * sizeof(unsigned char));
  _outA = (unsigned char *)malloc(num * (lenA + lenB) * sizeof(unsigned char));
  _outB = (unsigned char *)malloc(num * (lenA + lenB) * sizeof(unsigned char));

  // Fixed input
  for (int i = 0; i < 4; ++i) {
    _seqA[i] = 2;
  }
  for (int i = 4; i < lenA; ++i) {
    _seqA[i] = 1;
  }

  for (int i = 0; i < lenB; ++i) {
    _seqB[i] = 1;
  }
 
  // Duplicate for batching
  for (int i = 1; i < num; ++i) {
    memcpy(&_seqA[i * lenA], _seqA, sizeof(unsigned char) * lenA);
    memcpy(&_seqB[i * lenB], _seqB, sizeof(unsigned char) * lenB);
  }

#ifdef MCC_ACC
  __merlin_default_function(_seqA, _seqB, _outA, _outB);
#else
  default_function(_seqA, _seqB, _outA, _outB);
#endif

  // Verify result
  for (int i = 0; i < lenA + lenB; ++i) {
    if (i < 124 && _outA[i] != 1)
      cout << "Expect 1 at index " << i << " but " << _outA[i] << endl;
    else if (i >= 124 && _outA[i] != 0)
      cout << "Expect 0 at index " << i << " but " << _outA[i] << endl;
  }
  cout << "Result verified" << endl;

#ifdef MCC_ACC
  __merlin_release();
#endif

  return 0;
}
