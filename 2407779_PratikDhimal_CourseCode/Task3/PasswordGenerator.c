#include <stdio.h>
#include <stdlib.h>
#include <time.h>


void encrypt_ten_character(char* raw, char* out) {

    out[0] = raw[0] + 2;
    out[1] = raw[0] - 2;
    out[2] = raw[0] + 1;
    out[3] = raw[1] + 3;
    out[4] = raw[1] - 3;
    out[5] = raw[1] - 1;
    out[6] = raw[2] + 2;
    out[7] = raw[2] - 2;
    out[8] = raw[3] + 4;
    out[9] = raw[3] - 4;
    out[10] = '\0';

    // handle ascii wrapping
    for(int i=0; i<10; i++) {
        if(i < 6) {
            while(out[i] > 'z') out[i] -= 26;
            while(out[i] < 'a') out[i] += 26;
        } else {
            while(out[i] > '9') out[i] -= 10;
            while(out[i] < '0') out[i] += 10;
        }
    }
}

int main(int argc, char* argv[]) {
    if(argc != 2) {
        printf("Usage: %s <count (>= 10000)>\n", argv[0]);
        return 1;
    }

    int count = atoi(argv[1]);
    if(count < 10000) {
        printf("Error: count must be at least 10000\n");
        return 1;
    }

    FILE* fp = fopen("cpu_encrypted_password.txt", "w");

    // //just for checking purpose
    // FILE *ogPassword = fopen("originalPassword.txt", "w");

    if(!fp) {
        printf("Unable to open fiel for writing\n");
        return 1;
    }

    srand(time(NULL));
    char raw[5];
    char enc[12];

    for(int i=0; i<count; i++) {
        // generate raw passwords in format aa00, bb12, ce19
        raw[0] = 'a' + rand() % 26;
        raw[1] = 'a' + rand() % 26;
        raw[2] = '0' + rand() % 10;
        raw[3] = '0' + rand() % 10;
        raw[4] = '\0';
        
        // fprintf(ogPassword, "%s\n", raw);
    
        encrypt_ten_character(raw, enc);
        
        // save only the encrypted version
        fprintf(fp, "%s\n", enc);
    }

    fclose(fp);
    printf(" %d passwords generation succesfull to cpu_encrypted_password.txt \n", count);
    return 0;
}
