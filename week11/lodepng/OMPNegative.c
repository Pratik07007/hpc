#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char **argv){

    unsigned int error, encError;
    unsigned char* image;
    unsigned char* newImage;
    unsigned int width, height;

    const char* filename = argv[1];
    const char* newFileName = "generated.png";

    error = lodepng_decode32_file(&image, &width, &height, filename);
    if(error){
        printf("error %u: %s\n", error, lodepng_error_text(error));
        return 1;
    }

    newImage = malloc(width * height * 4);

    printf("width = %u height = %u\n", width, height);

    #pragma omp parallel for
    for(int i = 0; i < width * height * 4; i += 4){
        unsigned char r = image[i];
        unsigned char g = image[i + 1];
        unsigned char b = image[i + 2];
        unsigned char t = image[i + 3];

        newImage[i]     = 255 - r;
        newImage[i + 1] = 255 - g;
        newImage[i + 2] = 255 - b;
        newImage[i + 3] = t;
    }

    encError = lodepng_encode32_file(newFileName, newImage, width, height);
    if(encError){
        printf("error %u: %s\n", encError, lodepng_error_text(encError));
    }

    free(image);
    free(newImage);

    return 0;
}
