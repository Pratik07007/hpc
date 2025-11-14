/****************************************************************************
  This version adds 1000 coins to the account using 1000 threads. It uses
  a mutex to isolate a critical region of code.
  
  How long will it take to run?
  
  How much money will eventually be stored in the account?
  
  Compile with:

    cc -o penny08 penny08.c time_diff.c -lrt -pthread 

  Dr. Kevan Buckley, University of Wolverhampton, 2018 
*****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>
#include "time_diff.h"
#include <pthread.h>

pthread_mutex_t mutex;


void initialise_mutex() {
  int result =   pthread_mutex_init(&mutex, NULL);
  if(result != 0){
    printf("problem initialising mutex\n");
    exit(EXIT_FAILURE);
  }
}


int main(){
  struct timespec start, finish;
  long long int difference;   
  int account = 0;
  int i;

  int n = 1000;
  int result;

  initialise_mutex();

  clock_gettime(CLOCK_MONOTONIC, &start);

  void *add_penny();
  pthread_t t[n];
  for(i=0;i<n;i++){
    result = pthread_create(&t[i], NULL, add_penny, &account);
    if(result != 0){
      printf("thread %d failed to start\n", i);
      printf("try lowering stack size with ulimit -s 1024\n");
      printf("to find current limit: ulimit -a \n");
      exit(1);
    }
  }
  for(i=0;i<n;i++){
    pthread_join(t[i], NULL);
  }
  clock_gettime(CLOCK_MONOTONIC, &finish);
  time_difference(&start, &finish, &difference);
  printf("accumulated %dp\n", account);
  printf("run lasted %lldns or %9.5lfs\n", difference, difference/1000000000.0);

  return 0;
} 

void *add_penny(void *bal) {
  int *balance = bal;

  int result = pthread_mutex_lock(&mutex);
  if(result != 0){
    printf("problem locking mutex\n");
    exit(EXIT_FAILURE);
  }

  int b = *balance;
/* cause a short delay */
  int i;
  for(i=0;i<100000;i++){
  }
  b = b + 1;
  *balance = b;

  result = pthread_mutex_unlock(&mutex);
  if(result != 0){
    printf("problem unlocking mutex\n");
    exit(EXIT_FAILURE);
  }
}

