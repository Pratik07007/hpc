#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <omp.h>
#include <errno.h>
#include <stdint.h>

int check_numeric(const char *str)
{
    if (!str || !*str)
        return 0;
    char *endptr;
    errno = 0;
    strtod(str, &endptr);
    if (errno != 0 || endptr == str)
        return 0;
    while (*endptr)
    {
        if (!isspace((unsigned char)*endptr) && *endptr != ',')
            return 0;
        endptr++;
    }
    return 1;
}

int limit_threads(int requested, int max_dim_val)
{
    return (max_dim_val < 1) ? 1 : ((requested > max_dim_val) ? max_dim_val : requested);
}

int maximum(int x, int y) { return (x > y) ? x : y; }

double **allocate_matrix(int rows, int cols)
{
    if (rows <= 0 || cols <= 0)
        return NULL;
    double **mat = (double **)malloc(rows * sizeof(double *));
    if (!mat)
        return NULL;

    for (int i = 0; i < rows; i++)
    {
        mat[i] = (double *)malloc(cols * sizeof(double));
        if (!mat[i])
        {
            for (int k = 0; k < i; k++)
                free(mat[k]);
            free(mat);
            return NULL;
        }
    }
    return mat;
}

void deallocate_matrix(double **mat, int rows)
{
    if (!mat)
        return;
    for (int i = 0; i < rows; i++)
        if (mat[i])
            free(mat[i]);
    free(mat);
}

int read_token(FILE *fp, char *buf)
{
    if (fscanf(fp, " %127[^, \t\n]", buf) != 1)
        return 0;
    int ch = fgetc(fp);
    if (ch != ',' && !isspace(ch) && ch != EOF)
        ungetc(ch, fp);
    return 1;
}

double **load_matrix(FILE *fp, int *rows_out, int *cols_out)
{
    char token[128];
    if (!read_token(fp, token))
        return NULL;
    if (!check_numeric(token))
    {
        fprintf(stderr, "Rows header invalid: '%s'\n", token);
        return NULL;
    }
    int r = atoi(token);

    if (!read_token(fp, token))
        return NULL;
    if (!check_numeric(token))
    {
        fprintf(stderr, "Cols header invalid: '%s'\n", token);
        return NULL;
    }
    int c = atoi(token);

    if (r <= 0 || c <= 0)
    {
        fprintf(stderr, "Invalid matrix size %dx%d\n", r, c);
        return NULL;
    }

    double **M = allocate_matrix(r, c);
    if (!M)
    {
        fprintf(stderr, "Failed memory allocation for %dx%d\n", r, c);
        return NULL;
    }

    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++)
        {
            if (!read_token(fp, token))
            {
                fprintf(stderr, "Unexpected EOF while reading matrix data\n");
                deallocate_matrix(M, r);
                return NULL;
            }
            if (!check_numeric(token))
            {
                fprintf(stderr, "Non-numeric value detected: '%s'\n", token);
                deallocate_matrix(M, r);
                return NULL;
            }
            M[i][j] = strtod(token, NULL);
        }

    *rows_out = r;
    *cols_out = c;
    return M;
}

void display_matrix(FILE *fp, double **M, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            if (isnan(M[i][j]))
                fprintf(fp, "NaN");
            else
                fprintf(fp, "%0.6lf", M[i][j]);
            if (j < cols - 1)
                fprintf(fp, " | ");
        }
        fprintf(fp, "\n");
    }
}

double **matrix_add(double **A, double **B, int rows, int cols, int num_threads)
{
    double **R = allocate_matrix(rows, cols);
    if (!R)
        return NULL;
    int t = limit_threads(num_threads, maximum(rows, cols));
#pragma omp parallel for num_threads(t)
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            R[i][j] = A[i][j] + B[i][j];
    return R;
}

double **matrix_subtract(double **A, double **B, int rows, int cols, int num_threads)
{
    double **R = allocate_matrix(rows, cols);
    if (!R)
        return NULL;
    int t = limit_threads(num_threads, maximum(rows, cols));
#pragma omp parallel for num_threads(t)
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            R[i][j] = A[i][j] - B[i][j];
    return R;
}

double **matrix_elem_mul(double **A, double **B, int rows, int cols, int num_threads)
{
    double **R = allocate_matrix(rows, cols);
    if (!R)
        return NULL;
    int t = limit_threads(num_threads, maximum(rows, cols));
#pragma omp parallel for num_threads(t)
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            R[i][j] = A[i][j] * B[i][j];
    return R;
}

double **matrix_elem_div(double **A, double **B, int rows, int cols, int num_threads)
{
    double **R = allocate_matrix(rows, cols);
    if (!R)
        return NULL;
    int t = limit_threads(num_threads, maximum(rows, cols));
#pragma omp parallel for num_threads(t)
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            R[i][j] = (B[i][j] == 0.0) ? NAN : A[i][j] / B[i][j];
    return R;
}

double **matrix_transpose(double **A, int rows, int cols, int num_threads)
{
    double **T = allocate_matrix(cols, rows);
    if (!T)
        return NULL;
    int t = limit_threads(num_threads, rows);
#pragma omp parallel for num_threads(t)
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            T[j][i] = A[i][j];
    return T;
}

double **matrix_multiply(double **A, double **B, int rA, int cA, int cB, int num_threads)
{
    double **R = allocate_matrix(rA, cB);
    if (!R)
        return NULL;
    int t = limit_threads(num_threads, rA);
#pragma omp parallel for num_threads(t)
    for (int i = 0; i < rA; i++)
        for (int j = 0; j < cB; j++)
        {
            double sum = 0.0;
            for (int k = 0; k < cA; k++)
                sum += A[i][k] * B[k][j];
            R[i][j] = sum;
            
        }
    return R;
}

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        fprintf(stderr, "Usage: %s <input_file> <threads>\n", argv[0]);
        return 1;
    }

    char *endptr;
    long tcount = strtol(argv[2], &endptr, 10);
    if (*endptr != '\0' || tcount <= 0)
    {
        fprintf(stderr, "Invalid thread number.\n");
        return 1;
    }
    int threads = (int)tcount;

    FILE *fin = fopen(argv[1], "r");
    if (!fin)
    {
        fprintf(stderr, "Cannot open file %s\n", argv[1]);
        return 1;
    }

    FILE *fout = fopen("result.txt", "w");
    if (!fout)
    {
        fprintf(stderr, "Cannot create output file\n");
        fclose(fin);
        return 1;
    }

    int pair_idx = 0;

    while (1)
    {
        int rA, cA, rB, cB;
        double **A = load_matrix(fin, &rA, &cA);
        if (!A)
        {
            if (!feof(fin))
                fprintf(stderr, "Invalid Matrix A (pair %d)\n", pair_idx + 1);
            break;
        }

        double **B = load_matrix(fin, &rB, &cB);
        if (!B)
        {
            fprintf(stderr, "Invalid or missing Matrix B (pair %d)\n", pair_idx + 1);
            deallocate_matrix(A, rA);
            break;
        }

        pair_idx++;
        fprintf(fout, "\n########### MATRIX PAIR %d ###########\n\n", pair_idx);

        int shape_match = (rA == rB && cA == cB);
        double **R;

        if (shape_match)
        {
            fprintf(fout, "Sum Matrix (%dx%d):\n", rA, cA);
            R = matrix_add(A, B, rA, cA, threads);
            display_matrix(fout, R, rA, cA);
            deallocate_matrix(R, rA);

            fprintf(fout, "\nDifference Matrix (%dx%d):\n", rA, cA);
            R = matrix_subtract(A, B, rA, cA, threads);
            display_matrix(fout, R, rA, cA);
            deallocate_matrix(R, rA);

            fprintf(fout, "\nElement-wise Product (%dx%d):\n", rA, cA);
            R = matrix_elem_mul(A, B, rA, cA, threads);
            display_matrix(fout, R, rA, cA);
            deallocate_matrix(R, rA);

            fprintf(fout, "\nElement-wise Division (%dx%d):\n", rA, cA);
            R = matrix_elem_div(A, B, rA, cA, threads);
            display_matrix(fout, R, rA, cA);
            deallocate_matrix(R, rA);
        }
        else
        {
            fprintf(fout, "Cannot perform element-wise operations: shape mismatch.\n");
        }

        fprintf(fout, "\nTranspose of Matrix A (%dx%d):\n", cA, rA);
        R = matrix_transpose(A, rA, cA, threads);
        display_matrix(fout, R, cA, rA);
        deallocate_matrix(R, cA);

        fprintf(fout, "\nTranspose of Matrix B (%dx%d):\n", cB, rB);
        R = matrix_transpose(B, rB, cB, threads);
        display_matrix(fout, R, cB, rB);
        deallocate_matrix(R, cB);

        if (cA == rB)
        {
            fprintf(fout, "\nMatrix Multiplication A x B (%dx%d):\n", rA, cB);
            R = matrix_multiply(A, B, rA, cA, cB, threads);
            display_matrix(fout, R, rA, cB);
            deallocate_matrix(R, rA);
        }
        else
        {
            fprintf(fout, "\nMatrix multiplication not possible: columns of A != rows of B\n");
        }

        deallocate_matrix(A, rA);
        deallocate_matrix(B, rB);
    }

    fclose(fin);
    fclose(fout);
    return 0;
}
