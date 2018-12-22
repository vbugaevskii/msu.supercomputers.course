#include "utils.h"

__global__ void cuda_set_slice_x(double *data, cell_info_t params, int i, double value)
{
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;

    if (j >= params.dy || k >= params.dz)
        return;

    data[index(i, j, k, params)] = value;
}

__global__ void cuda_set_slice_y(double *data, cell_info_t params, int j, double value)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int k = blockDim.z * blockIdx.z + threadIdx.z;

    if (i >= params.dx || k >= params.dz)
        return;

    data[index(i, j, k, params)] = value;
}

__global__ void cuda_set_slice_z(double *data, cell_info_t params, int k, double value)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i >= params.dx || j >= params.dy)
        return;

    data[index(i, j, k, params)] = value;
}

__global__ void cuda_copy_slice_x(double *data, cell_info_t params, int i_dst, int i_src)
{
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;

    if (j >= params.dy || k >= params.dz)
        return;

    data[index(i_dst, j, k, params)] = data[index(i_src, j, k, params)];
}

__global__ void cuda_buffer_slice_copy_x(double *buffer, double *data, cell_info_t params, int i, bool get)
{
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;

    if (j >= params.dy || k >= params.dz)
        return;

    int p = j + k * params.dy;

    if (get)
        buffer[p] = data[index(i, j, k, params)];
    else
        data[index(i, j, k, params)] = buffer[p];
}

__global__ void cuda_copy_slice_y(double *data, cell_info_t params, int j_dst, int j_src)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int k = blockDim.z * blockIdx.z + threadIdx.z;

    if (i >= params.dx || k >= params.dz)
        return;

    data[index(i, j_dst, k, params)] = data[index(i, j_src, k, params)];
}

__global__ void cuda_buffer_slice_copy_y(double *buffer, double *data, cell_info_t params, int j, bool get)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int k = blockDim.z * blockIdx.z + threadIdx.z;

    if (i >= params.dx || k >= params.dz)
        return;

    int p = i + k * params.dx;

    if (get)
        buffer[p] = data[index(i, j, k, params)];
    else
        data[index(i, j, k, params)] = buffer[p];
}

__global__ void cuda_copy_slice_z(double *data, cell_info_t params, int k_dst, int k_src)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i >= params.dx || j >= params.dy)
        return;

    data[index(i, j, k_dst, params)] = data[index(i, j, k_src, params)];
}

__global__ void cuda_buffer_slice_copy_z(double *buffer, double *data, cell_info_t params, int k, bool get)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i >= params.dx || j >= params.dy)
        return;

    int p = i + j * params.dx;

    if (get)
        buffer[p] = data[index(i, j, k, params)];
    else
        data[index(i, j, k, params)] = buffer[p];
}