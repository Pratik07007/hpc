#include <stdio.h>

void main()

{

FILE *fptr;
char line [1000];
fptr = fopen("file38_g7.txt", "r");


while(!feof(fptr)){

fgets(line,10,fptr);
printf("Line  %s \n", line);
}
fclose(fptr);

}
