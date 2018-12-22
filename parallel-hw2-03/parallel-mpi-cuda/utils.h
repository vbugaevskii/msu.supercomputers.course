#ifndef PARALLEL_MPI_UTILS_H
#define PARALLEL_MPI_UTILS_H

#include <mpi.h>
#include <cuda.h>

#include "timer.h"

struct TimersArray {
    MPITimer total;
    MPITimer init;
    MPITimer free;
    MPITimer sendrecv;
    double copy;
};

extern int rank;

extern int N, K;
extern int dx, dy, dz;
extern double hx, hy, hz;

#define ndim 3

struct cell_info_t {
    int dx, dy, dz;
    double hx, hy, hz, tau;
    int i_min, j_min, k_min;
    char fl_mask;
    char bc_mask;
};

int split(int N, int n_threads) {
    return ceil(double(N) / n_threads);
}

__device__ int index(int i, int j, int k, cell_info_t params) {
    return i + j * params.dx + k * params.dx * params.dy;
}

__device__ double laplace(const double *data, int i, int j, int k, cell_info_t params)
{
    double val = 0;
    int i_prev, i_curr = index(i, j, k, params), i_next;

    i_prev = index(i-1, j, k, params);
    i_next = index(i+1, j, k, params);
    val += (data[i_prev] - 2 * data[i_curr] + data[i_next]) / params.hx / params.hx;

    i_prev = index(i, j-1, k, params);
    i_next = index(i, j+1, k, params);
    val += (data[i_prev] - 2 * data[i_curr] + data[i_next]) / params.hy / params.hy;

    i_prev = index(i, j, k-1, params);
    i_next = index(i, j, k+1, params);
    val += (data[i_prev] - 2 * data[i_curr] + data[i_next]) / params.hz / params.hz;

    return val;
}

char pack_fl_mask(const bool *is_first, const bool *is_last)
{
    char fl_mask = 0;
    for (char d = 0; d < ndim; d++)
        fl_mask |= ( char(is_first[d]) << 1 | char(is_last[d]) ) << (2 * d);
    return fl_mask;
}

__host__ __device__ void unpack_fl_mask(bool *is_first, bool *is_last, char fl_mask)
{
    for (char d = 0; d < ndim; d++) {
        is_first[d] = fl_mask & 2;
        is_last[d] = fl_mask & 1;
        fl_mask >>= 2;
    }
}

char pack_bc_mask(const char *bc) {
    char bc_mask;
    for (char d = 0; d < ndim; d++)
        bc_mask |= char(bc[d] == 2) << d;
    return bc_mask;
}

__host__ __device__ void unpack_bc_mask(char *bc, char bc_mask) {
    for (char d = 0; d < ndim; d++) {
        bc[d] = 1 + (bc_mask & 1);
        bc_mask >>= 1;
    }
}

#endif //PARALLEL_MPI_UTILS_H
