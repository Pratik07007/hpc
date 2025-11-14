#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
    for (int n = 1; n <= 100; n++) {
        if (n % 2 == 0 && n % 3 == 0 && n % 5 == 0) {
            printf("BishBashBosh\n");
        } else if (n % 2 == 0 && n % 3 == 0) {
            printf("BishBash\n");
        } else if (n % 2 == 0 && n % 5 == 0) {
            printf("BishBosh\n");
        } else if (n % 3 == 0 && n % 5 == 0) {
            printf("BashBosh\n");
        } else if (n % 2 == 0) {
            printf("Bish\n");
        } else if (n % 3 == 0) {
            printf("Bash\n");
        } else if (n % 5 == 0) {
            printf("Bosh\n");
        } else {
            printf("%d\n", n);
        }
    }

    return 0;
}
