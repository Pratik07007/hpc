#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{

    char *name = argv[1];
    int age = atoi(argv[2]);

    printf("Hello %s, you are %d years old.\n", name, age);

    return 0;
}
