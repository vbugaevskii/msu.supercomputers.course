#ifndef PARALLEL_MPI_BORDER_CONDITIONS_H
#define PARALLEL_MPI_BORDER_CONDITIONS_H

#include "utils.h"

void border_condition_1st_x(double *data, bool is_first, bool is_last)
{
    if (!is_first && !is_last)
        return;

#pragma omp parallel for
    for (int p = 0; p < dy * dz; p++) {
        // x_{0} = x_{N} = 0

        int j = p % dy;
        int k = (p / dy) % dz;

        if (is_first)
            data[index(1, j, k)] = 0;

        if (is_last)
            data[index(dx - 2, j, k)] = 0;
    }
}

void border_condition_1st_y(double *data, bool is_first, bool is_last)
{
    if (!is_first && !is_last)
        return;

#pragma omp parallel for
    for (int p = 0; p < dx * dz; p++) {
        // y_{0} = y_{N} = 0

        int i = p % dx;
        int k = (p / dx) % dz;

        if (is_first)
            data[index(i, 1, k)] = 0;

        if (is_last)
            data[index(i, dy - 2, k)] = 0;
    }
}

void border_condition_1st_z(double *data, bool is_first, bool is_last)
{
    if (!is_first && !is_last)
        return;

#pragma omp parallel for
    for (int p = 0; p < dx * dy; p++) {
        // z_{0} = z_{N} = 0

        int i = p % dx;
        int j = (p / dx) % dy;

        if (is_first)
            data[index(i, j, 1)] = 0;

        if (is_last)
            data[index(i, j, dz - 2)] = 0;
    }
}

void border_condition_2nd_x(double *data, bool is_first, bool is_last)
{
#pragma omp parallel for
    for (int p = 0; p < dy * dz; p++) {
        // x_{0} = x_{N-1}; x_{N} = x_{1}

        int j = p % dy;
        int k = (p / dy) % dz;

        /*
         * ЗАМЕЧАНИЕ:
         *  x_{N} хранится в "левой" обменной области;
         *  x_{1} хранится в "правой" обменной области.
         */

        if (is_first)
            data[index(1, j, k)] = data[index(0, j, k)];
    }
}

void border_condition_2nd_y(double *data, bool is_first, bool is_last)
{
#pragma omp parallel for
    for (int p = 0; p < dx * dz; p++) {
        // y_{0} = y_{N-1}; y_{N} = y_{1}

        int i = p % dx;
        int k = (p / dx) % dz;

        /*
         * ЗАМЕЧАНИЕ:
         *  y_{N} хранится в "левой" обменной области;
         *  y_{1} хранится в "правой" обменной области.
         */

        if (is_first)
            data[index(i, 1, k)] = data[index(i, 0, k)];
    }
}

void border_condition_2nd_z(double *data, bool is_first, bool is_last)
{
#pragma omp parallel for
    for (int p = 0; p < dx * dy; p++) {
        // z_{0} = z_{N-1}; z_{N} = z_{1}

        int i = p % dx;
        int j = (p / dx) % dy;

        /*
         * ЗАМЕЧАНИЕ:
         *  y_{N} хранится в "левой" обменной области;
         *  y_{1} хранится в "правой" обменной области.
         */

        if (is_first)
            data[index(i, j, 1)] = data[index(i, j, 0)];
    }
}

#endif //PARALLEL_MPI_BORDER_CONDITIONS_H
