#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* simple math encryption
   input  : 4 chars
   output : 10 chars + '\0'
*/

void genPassword(char *raw, char *out)
{
    out[0]  = raw[0] + 2;
    out[1]  = raw[0] - 2;
    out[2]  = raw[0] + 1;

    out[3]  = raw[1] + 3;
    out[4]  = raw[1] - 3;
    out[5]  = raw[1] - 1;

    out[6]  = raw[2] + 2;
    out[7]  = raw[2] - 2;

    out[8]  = raw[3] + 4;
    out[9]  = raw[3] - 4;

    out[10] = '\0';

    for (int i = 0; i < 10; ++i)
    {
        if (i < 6)
        {
            while (out[i] > 'z') out[i] -= 26;
            while (out[i] < 'a') out[i] += 26;
        }
        else
        {
            while (out[i] > '9') out[i] -= 10;
            while (out[i] < '0') out[i] += 10;
        }
    }
}

int main(void)
{
    int count;

    printf("Enter number of passwords to generate (>= 10000): ");
    scanf("%d", &count);

    if (count < 10000)
    {
        printf("Error: count must be at least 10000\n");
        return 1;
    }

    FILE *fp = fopen("passwords.txt", "w");
    if (fp == NULL)
    {
        printf("file error\n");
        return 1;
    }

    srand(time(NULL));

    char raw[5];
    char enc[12];

    for (int i = 0; i < count; ++i)
    {
        raw[0] = (char)('a' + rand() % 26);
        raw[1] = (char)('a' + rand() % 26);
        raw[2] = (char)('0' + rand() % 10);
        raw[3] = (char)('0' + rand() % 10);
        raw[4] = '\0';

        genPassword(raw, enc);
        fprintf(fp, "%s\n", enc);
    }

    fclose(fp);

    printf("generated %d passwords to passwords.txt\n", count);
    return 0;
}
