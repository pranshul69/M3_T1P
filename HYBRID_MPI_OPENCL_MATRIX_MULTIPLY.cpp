#include <mpi.h>
#include <CL/cl.hpp>
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

void matrix_multiply(int *a, int *b, int *c, int chunk_size, cl::Context context, cl::CommandQueue queue, cl::Program program) {
    cl_int err;
    cl::Kernel kernel(program, "matrix_multiply", &err);
    if (err != CL_SUCCESS) {
        cerr << "Error creating kernel: " << err << endl;
        exit(1);
    }

    cl::Buffer buf_a(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * N * chunk_size, a, &err);
    if (err != CL_SUCCESS) {
        cerr << "Error creating buffer for A: " << err << endl;
        exit(1);
    }

    cl::Buffer buf_b(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * N * N, b, &err);
    if (err != CL_SUCCESS) {
        cerr << "Error creating buffer for B: " << err << endl;
        exit(1);
    }

    cl::Buffer buf_c(context, CL_MEM_WRITE_ONLY, sizeof(int) * N * chunk_size, NULL, &err);
    if (err != CL_SUCCESS) {
        cerr << "Error creating buffer for C: " << err << endl;
        exit(1);
    }

    kernel.setArg(0, buf_a);
    kernel.setArg(1, buf_b);
    kernel.setArg(2, buf_c);
    kernel.setArg(3, N);

    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(N, chunk_size), cl::NullRange);
    queue.enqueueReadBuffer(buf_c, CL_TRUE, 0, sizeof(int) * N * chunk_size, c);
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
    MPI_Scatter(a, chunk_size * N, MPI_INT, chunk_a, chunk_size * N, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(b, N * N, MPI_INT, 0, MPI_COMM_WORLD);
cl::Platform platform;
cl::Device device;
cl::Context context;
cl::CommandQueue queue;
cl::Program program;
cl_int err;

// Find platform and device
vector<cl::Platform> platforms;
vector<cl::Device> devices;
cl::Platform::get(&platforms);
platform = platforms[0];
platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
device = devices[0];

// Create context and queue
context = cl::Context(device);
queue = cl::CommandQueue(context, device);

// Read kernel file
ifstream file("matrix_multiply_kernel.cl");
string code(istreambuf_iterator<char>(file), (istreambuf_iterator<char>()));
cl::Program::Sources sources(1, make_pair(code.c_str(), code.length() + 1));

// Create program and build it
program = cl::Program(context, sources);
program.build({device});

// Multiply matrices
matrix_multiply(chunk_a, b, chunk_c, chunk_size, context, queue, program);

MPI_Gather(chunk_c, chunk_size * N, MPI_INT, c, chunk_size * N, MPI_INT, 0, MPI_COMM_WORLD);

if (rank == 0) {
    cout << "Matrix C:" << endl;
    print_matrix(c);
}

MPI_Finalize();
return 0;
}
