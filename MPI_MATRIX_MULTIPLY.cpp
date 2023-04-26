#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;

const int N = 1000; // size of matrices

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

void matrix_multiply(int *a, int *b, int *c, int chunk_size, int rank) {
    int *buffer = new int[N * chunk_size];
    MPI_Bcast(b, N * N, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(a, N * chunk_size, MPI_INT, buffer, N * chunk_size, MPI_INT, 0, MPI_COMM_WORLD);
    for (int i = 0; i < chunk_size; i++) {
        for (int j = 0; j < N; j++) {
            int sum = 0;
            for (int k = 0; k < N; k++) {
                sum += buffer[i * N + k] * b[k * N + j];
            }
            c[i * N + j + rank * chunk_size * N] = sum;
        }
    }
    delete[] buffer;
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
        
    }

    chunk_size = N / size;
    MPI_Bcast(&chunk_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    matrix_multiply(a, b, c, chunk_size, rank);

    int *result = new int[N * N];
    MPI_Gather(c, N * chunk_size, MPI_INT, result, N * chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        cout << "Result Matrix:" << endl;
        print_matrix(result);
    }

    delete[] a;
    delete[] b;
    delete[] c;
    delete[] result;

    MPI_Finalize();

    return 0;
}
