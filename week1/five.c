#include <stdio.h>
#include <stdlib.h>

void main()
{
    int sizeOfArray;
    printf("Enter the size of the array: ");
    scanf("%d", &sizeOfArray);

    int *array = (int *)malloc(sizeof(int) * (sizeOfArray));
    for (int i = 0; i < sizeOfArray; i++)
    {
        printf("Enter the element %d: ", i);
        scanf("%d", &array[i]);
    }

    printf("The array is: ");
    for (int i = 0; i < sizeOfArray; i++)
    {
        printf("%d ", array[i]);
    }
}