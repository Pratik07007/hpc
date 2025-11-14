#include <stdio.h>
#include <stdlib.h>

int main()
{

int n;
int *p;
n=3;
p=&n;
printf("The value of n is %d \n",n);
printf("The value of p is %d \n",*p);
printf("The memory address is %p \n",p);
return 0;

}

