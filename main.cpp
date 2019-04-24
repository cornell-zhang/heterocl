#include <sys/ipc.h>
#include <sys/shm.h>
int main(void) { 
  int* A= (int*)shmat(3440640, (void*)0, 0);
  A[0] = 40;
}
