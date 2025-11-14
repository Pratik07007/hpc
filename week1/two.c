#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
    char name[100];  
    int age;

    printf("Please Enter your name: \n");
    scanf("%99s", name);  // limit input to prevent overflow
    printf("Please Enter your age: \n");
    scanf("%d", &age);

    printf("Hello %s, you are %d years old.\n", name, age);

    return 0;
}
