#include <stdio.h>

void main(){
FILE *fptr;
fptr= fopen("file38.txt","w");
fprintf(fptr,"Number uis written with hpc group 7 %d \n",123);
fclose(fptr);
}
