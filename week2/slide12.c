#include <stdio.h>

int main(){
int a[10], *p,*q;
p=&a[2];
q=p+3;
p++;
p--;
*p=123;
*q=*p;
q=p;
scanf("%d",q);
printf("a[4] is %d",a[4]);
return 0;;



}
