#include <stdio.h>

int main(){
int a[10],*p;
p=&a;
p[0]=10;
p[1]=10;
printf("%d",p[1]);
return 0;
}
