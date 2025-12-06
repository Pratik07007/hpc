#include<stdio.h>
#include<omp.h>

int counter=0;
omp_lock_t lock;

int doSomething(int threadID,  int count)
{
	printf("Thread %d: COunter=%d\n", threadID, counter);
	return count + 1;
}

void myThread()
{
	int tid = omp_get_thread_num();
	for(int i = 0; i<100; i++)
	{
		omp_set_lock(&lock);
		counter = doSomething(tid, counter);
		omp_unset_lock(&lock);
	}
}

void main()
{
	omp_init_lock(&lock);
	#pragma omp parallel
	myThread();
	omp_destroy_lock(&lock);
	printf("Final counter = %d\n", counter);
}
