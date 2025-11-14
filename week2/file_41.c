#include <stdio.h>

void main()

{

char c;

FILE *fptr;

fptr = fopen("file38_g7.txt", "r");

c = fgetc(fptr);

while(c!=EOF){

printf("%c", c);
 c = fgetc(fptr);
}
fclose(fptr);

}
