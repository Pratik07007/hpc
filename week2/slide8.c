#include <stdio.h>

int main(){
int n=3;
 n=100;
int *p;
p = &n;
n++;
printf("Now n is %d \n",*p);
p=&n;
printf("n is %d \n",*p);
*p=500;
printf("now n is %d \n",n);
return 0;


}
