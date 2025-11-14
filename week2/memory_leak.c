#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MB_5 (1024 * 1024 * 1024) // Define 5 MB in bytes

void memoryLeak() {
    // Dynamically allocate 5 MB of memory
    char *ptr = (char *)malloc(MB_5);

    if (ptr == NULL) {
        printf("Memory allocation failed!\n");
        return;
    }

    // Optionally, you can initialize the memory with some data
    memset(ptr, 0, MB_5);

    // We are intentionally not freeing the allocated memory to simulate a memory leak
    // free(ptr); // This line should be here to free the allocated memory, but it's intentionally left out
}

int main() {
    char response[4]; // To store user input

    while (1) {
        printf("Do you want to continue allocating memory? Type 'yes' to continue or 'no' to stop: ");
        scanf("%s", response);

        // Convert user response to lowercase for easier comparison
        if (strcmp(response, "no") == 0) {
            printf("Stopping the memory allocation loop.\n");
            break; // Exit the loop if the user types "no"
        } else if (strcmp(response, "yes") == 0) {
            memoryLeak();
        } else {
            printf("Invalid input! Please type 'yes' or 'no'.\n");
        }
    }

    return 0;
}
