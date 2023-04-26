#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;

const int N = 2000; // size of matrices

void fill_matrix(int *matrix) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i * N + j] = rand() % 100;
        }
    }
}

void print_matrix(int *matrix) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cout << matrix[i * N + j] << " ";
        }
        cout << endl;
    }
}

void matrix_multiply(int *a, int *b, int *c, int chunk_size) {
    #pragma omp parallel for shared(a, b, c) schedule(static)
    for (int i = 0; i < chunk_size; i++) {
        for (int j = 0; j < N; j++) {
            int sum = 0;
            for (int k = 0; k < N; k++) {
                sum += a[i * N + k] * b[k * N + j];
            }
            c[i * N + j] = sum;
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    int *a = new int[N * N];
    int *b = new int[N * N];
    int *c = new int[N * N];
    int chunk_size;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        fill_matrix(a);
        fill_matrix(b);
        cout << "Matrix A:" << endl;
        print_matrix(a);
        cout << "Matrix B:" << endl;
        print_matrix(b);
    }

    chunk_size = N / size;
    int *chunk_a = new int[N * chunk_size];
    int *chunk_c = new int[N * chunk_size];
    MPI_Scatter(a, N * chunk_size, MPI_INT, chunk_a, N * chunk_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(b, N * N, MPI_INT, 0, MPI_COMM_WORLD);

    matrix_multiply(chunk_a, b, chunk_c, chunk_size);

    MPI_Gather(chunk_c, N * chunk_size, MPI_INT, c, N * chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        cout << "Result Matrix:" << endl;
        print_matrix(c);
    }

    delete[] a;
    delete[] b;
    delete[] c;
    delete[] chunk_a;
    delete[] chunk_c;

    MPI_Finalize();

    return 0;
}
