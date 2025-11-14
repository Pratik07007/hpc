/****************************************************************************
  This version does the same as the previous version, but records how long
  it takes to run. This is so that the relative speeds of later versions can 
  be compared.
  
  Compile with:

    cc -o penny02 penny02.c time_diff.c -lrt

  Dr. Kevan Buckley, University of Wolverhampton, 2018
*****************************************************************************/

#include <stdio.h>
#include <unistd.h>
#include "time_diff.h"

void add_penny(int *balance) {
  int b = *balance;
  usleep(1000000);
  b = b + 1;
  *balance = b;
}

int main(){
  struct timespec start, finish;
  long long int difference;   
  int account = 0;
  clock_gettime(CLOCK_MONOTONIC, &start);

  add_penny(&account);

  clock_gettime(CLOCK_MONOTONIC, &finish);
  time_difference(&start, &finish, &difference);
  printf("accumulated %dp\n", account);
  printf("run lasted %9.5lfs\n", difference/1000000000.0);
  return 0;
} 
