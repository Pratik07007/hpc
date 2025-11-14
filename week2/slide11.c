#include <stdio.h>

int main(){

int a[10], *p;
p=&a[2];
*p=10;
*(p+1)=10;
*(p+3)=12;
printf("\n %d",*(p+3));
printf("\n %d",*(p+5));
printf("\n %d",*(p+4));
return 0;
}
