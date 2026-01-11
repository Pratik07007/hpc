#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <crypt.h>

// standard sha512 salt
#define SALT "$6$AS$"

int main(int argc, char* argv[]) {
    if(argc != 3) {
        printf("usage: ./hasher <input_file> <output_file>\n");
        return 1;
    }

    FILE* fin = fopen(argv[1], "r");
    FILE* fout = fopen(argv[2], "w");
    
    if(!fin || !fout) { printf("file open error\n"); return 1; }

    char line[100];
    // read every line
    while(fgets(line, sizeof(line), fin)) {
        // strip newline calls
        line[strcspn(line, "\r\n")] = 0;
        
        // hash it using linux crypt lib
        char* hash = crypt(line, SALT);
        
        // save hash
        fprintf(fout, "%s\n", hash);
    }

    fclose(fin);
    fclose(fout);
    printf("hashing done from %s to %s\n", argv[1], argv[2]);
    return 0;
}