#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <ctype.h>

typedef struct {
    char word[50];
    int count;
} WordEntry;

WordEntry words[5000];
int word_count = 0;

pthread_mutex_t lock;

typedef struct {
    char *text_buffer;
    long start_index;
    long end_index;
    long buffer_size;
} ThreadArgs;

void add_word(char *w) {
    pthread_mutex_lock(&lock);
    for (int i = 0; i < word_count; i++) {
        if (strcmp(words[i].word, w) == 0) {
            words[i].count++;
            pthread_mutex_unlock(&lock);
            return;
        }
    }
    if (word_count < 5000) {
        strcpy(words[word_count].word, w);
        words[word_count].count = 1;
        word_count++;
    }
    pthread_mutex_unlock(&lock);
}

void *process_slice(void *arg) {
    ThreadArgs *data = (ThreadArgs *)arg;
    if (data == NULL) return NULL;

    char current_word[50];
    int pos = 0;

    long i = data->start_index;
    
    // This line Skips partial word at the start (if we're in the middle of a word)
    if (i > 0 && isalnum(data->text_buffer[i-1])) {
        while (i < data->end_index && i < data->buffer_size && isalnum(data->text_buffer[i])) {
            i++;
        }
    }

    // Process words in this slice
    for (; i < data->end_index && i < data->buffer_size; i++) {
        char c = data->text_buffer[i];
        if (isalnum(c)) {
            if (pos < 49) {
                current_word[pos++] = tolower(c);
            }
        } else {
            if (pos > 0) {
                current_word[pos] = '\0';
                add_word(current_word);
                pos = 0;
            }
        }
    }

    // Complete the word that crosses the boundary
    while (i < data->buffer_size && isalnum(data->text_buffer[i])) {
        if (pos < 49) {
            current_word[pos++] = tolower(data->text_buffer[i]);
        }
        i++;
    }

    if (pos > 0) {
        current_word[pos] = '\0';
        add_word(current_word);
    }
    
    return NULL;
}

// Comparison function for qsort (frequency descending order)
int compare_frequency(const void *a, const void *b) {
    WordEntry *wa = (WordEntry *)a;
    WordEntry *wb = (WordEntry *)b;
    return wb->count - wa->count; // Descending order
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <filename> <thread_count>\n", argv[0]);
        return 1;
    }

    char *endptr;
    long thread_count = strtol(argv[2], &endptr, 10);
    if (*endptr != '\0' || thread_count < 1) {
        printf("Please enter a valid positive integer for the thread count.\n");
        return 1;
    }

    FILE *file = fopen(argv[1], "r");
    if (file == NULL) {
        perror("Error opening input file");
        return 1;
    }

    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    rewind(file);

    if (file_size == 0) {
        printf("Error: File is empty\n");
        fclose(file);
        return 1;
    }

    char *buffer = malloc(file_size + 1);
    if (buffer == NULL) {
        perror("Error allocating memory");
        fclose(file);
        return 1;
    }

    size_t bytes_read = fread(buffer, 1, file_size, file);
    if (bytes_read == 0) {
        perror("Error reading file");
        free(buffer);
        fclose(file);
        return 1;
    }
    buffer[bytes_read] = '\0';
    fclose(file);

    long actual_size = bytes_read;

    pthread_t *threads = malloc(thread_count * sizeof(pthread_t));
    ThreadArgs *thread_args = malloc(thread_count * sizeof(ThreadArgs));
    
    if (threads == NULL || thread_args == NULL) {
        perror("Error allocating memory for threads");
        free(buffer);
        free(threads);
        free(thread_args);
        return 1;
    }

    long slice_size = actual_size / thread_count;

    pthread_mutex_init(&lock, NULL);

    printf("Processing file with %ld threads...\n", thread_count);
    printf("File size: %ld bytes\n", actual_size);
    printf("Slice size: %ld bytes per thread\n\n", slice_size);

    for (int i = 0; i < thread_count; i++) {
        thread_args[i].text_buffer = buffer;
        thread_args[i].start_index = i * slice_size;
        thread_args[i].end_index = (i == thread_count - 1) ? actual_size : (i + 1) * slice_size;
        thread_args[i].buffer_size = actual_size;
        
        if (pthread_create(&threads[i], NULL, process_slice, &thread_args[i]) != 0) {
            perror("Failed to create thread");
            free(buffer);
            free(threads);
            free(thread_args);
            return 1;
        }
    }

    for (int i = 0; i < thread_count; i++) {
        pthread_join(threads[i], NULL);
    }

    pthread_mutex_destroy(&lock);

    // Sort words by frequency (descending order)
    qsort(words, word_count, sizeof(WordEntry), compare_frequency);

    FILE *out = fopen("result.txt", "w");
    if (out == NULL) {
        perror("Error opening output file");
        free(buffer);
        free(threads);
        free(thread_args);
        return 1;
    }

    fprintf(out, "Word Frequency Count Results\n");
    fprintf(out, "=============================\n");
    fprintf(out, "Total unique words: %d\n", word_count);
    fprintf(out, "(Sorted by frequency - descending order)\n\n");
    fprintf(out, "%-20s %s\n", "Word", "Count");
    fprintf(out, "-------------------- ------\n");

    int total_words = 0;
    for (int i = 0; i < word_count; i++) {
        fprintf(out, "%-20s %6d\n", words[i].word, words[i].count);
        total_words += words[i].count;
    }

    fprintf(out, "\n=============================\n");
    printf("Processing complete!\n");
    printf("Total unique words: %d\n", word_count);
    printf("Results saved to result.txt\n");

    fclose(out);
    free(buffer);
    free(threads);
    free(thread_args);
    
    return 0;
}