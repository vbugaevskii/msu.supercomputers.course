#ifndef PARALLEL_MPI_UTILS_H
#define PARALLEL_MPI_UTILS_H

#include <mpi.h>
#include <omp.h>

#include "timer.h"

struct TimersArray {
    ChronoTimer total;
    ChronoTimer init;
    ChronoTimer free;
    MPITimer sendrecv;
    ChronoTimer copy;
};

extern int dx, dy, dz;
extern double hx, hy, hz;

inline int index(int i, int j, int k) {
    return i + j * dx + k * dx * dy;
}

inline double laplace(const double data[], int i, int j, int k) {
    double val = 0;
    int p_prev, p_curr = index(i, j, k), p_next;

    p_prev = index(i-1, j, k);
    p_next = index(i+1, j, k);
    val += (data[p_prev] - 2 * data[p_curr] + data[p_next]) / hx / hx;

    p_prev = index(i, j-1, k);
    p_next = index(i, j+1, k);
    val += (data[p_prev] - 2 * data[p_curr] + data[p_next]) / hy / hy;

    p_prev = index(i, j, k-1);
    p_next = index(i, j, k+1);
    val += (data[p_prev] - 2 * data[p_curr] + data[p_next]) / hz / hz;

    return val;
}

#endif //PARALLEL_MPI_UTILS_H
