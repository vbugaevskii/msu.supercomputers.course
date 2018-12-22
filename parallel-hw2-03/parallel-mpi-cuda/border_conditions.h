#ifndef PARALLEL_MPI_BORDER_CONDITIONS_H
#define PARALLEL_MPI_BORDER_CONDITIONS_H

#include "utils.h"
#include "slice_operations.h"

extern cell_info_t params;

void border_condition_1st_x(double *data, bool is_first, bool is_last)
{
    if (!is_first && !is_last)
        return;

    dim3 threads(1, 16, 16);
    dim3 blocks(1, split(dy, threads.y), split(dz, threads.z));

    if (is_first)
        cuda_set_slice_x <<< blocks, threads >>> (data, params, 1, 0.0);

    if (is_last)
        cuda_set_slice_x <<< blocks, threads >>> (data, params, dx - 2, 0.0);
}

void border_condition_1st_y(double *data, bool is_first, bool is_last)
{
    if (!is_first && !is_last)
        return;

    dim3 threads(16, 1, 16);
    dim3 blocks(split(dx, threads.x), 1, split(dz, threads.z));

    if (is_first)
        cuda_set_slice_y <<< blocks, threads >>> (data, params, 1, 0.0);

    if (is_last)
        cuda_set_slice_y <<< blocks, threads >>> (data, params, dy - 2, 0.0);
}

void border_condition_1st_z(double *data, bool is_first, bool is_last)
{
    if (!is_first && !is_last)
        return;

    dim3 threads(16, 16, 1);
    dim3 blocks(split(dx, threads.x), split(dy, threads.y), 1);

    if (is_first)
        cuda_set_slice_z <<< blocks, threads >>> (data, params, 1, 0.0);

    if (is_last)
        cuda_set_slice_z <<< blocks, threads >>> (data, params, dz - 2, 0.0);
}

void border_condition_2nd_x(double *data, bool is_first, bool is_last)
{
    if (!is_first && !is_last)
        return;

    dim3 threads(1, 16, 16);
    dim3 blocks(1, split(dy, threads.y), split(dz, threads.z));

    if (is_first)
        cuda_copy_slice_x <<< blocks, threads >>> (data, params, 1, 0);
}

void border_condition_2nd_y(double *data, bool is_first, bool is_last)
{
    if (!is_first && !is_last)
        return;

    dim3 threads(16, 1, 16);
    dim3 blocks(split(dx, threads.x), 1, split(dz, threads.z));

    if (is_first)
        cuda_copy_slice_y <<< blocks, threads >>> (data, params, 1, 0);
}

void border_condition_2nd_z(double *data, bool is_first, bool is_last)
{
    if (!is_first && !is_last)
        return;

    dim3 threads(16, 16, 1);
    dim3 blocks(split(dx, threads.x), split(dy, threads.y), 1);

    if (is_first)
        cuda_copy_slice_z <<< blocks, threads >>> (data, params, 1, 0);
}

#endif //PARALLEL_MPI_BORDER_CONDITIONS_H
