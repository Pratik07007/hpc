/****************************************************************************
  This version adds a coin to the account using a thread.
  
  Compile with:

    cc -o penny04 penny04.c time_diff.c -lrt -pthread 

  Dr. Kevan Buckley, University of Wolverhampton, 2018
*****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "time_diff.h"
#include <pthread.h>
#include <unistd.h>

void *add_penny(void *balance) {
  int *b = balance;
  int c = *b;

// 1 second delay (simulating large calculation time)

  usleep(1000000);

  c = c + 1;
  *b = c;
}

int main(){
  struct timespec start, finish;
  long long int difference;   
  int account = 0;

  clock_gettime(CLOCK_MONOTONIC, &start);

  pthread_t t;

  /* start a thread to call the add_penny function */
  void *add_penny();
  pthread_create(&t, NULL, add_penny, &account);

  /* wait for the thread to finish*/
  pthread_join(t, NULL);
  
  clock_gettime(CLOCK_MONOTONIC, &finish);
  time_difference(&start, &finish, &difference);
  printf("accumulated %dp\n", account);
  printf("run lasted %lldns or %9.5lfs\n", difference, difference/1000000000.0);
  return 0;
} 
