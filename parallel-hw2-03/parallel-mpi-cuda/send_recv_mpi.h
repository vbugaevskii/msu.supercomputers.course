#ifndef PARALLEL_MPI_SEND_RECV_MPI_H
#define PARALLEL_MPI_SEND_RECV_MPI_H

#include "utils.h"

extern TimersArray timers;

extern cell_info_t params;

void send_recv_forward_x(double *data, MPI_Comm& comm_cart, int rank_prev, int rank_next, bool is_first, bool is_last)
{
    cudaError_t err;

    const int dim1 = dy;
    const int dim2 = dz;
    const int size = dim1 * dim2;

    dim3 threads(1, 16, 16);
    dim3 blocks(1, split(dy, threads.y), split(dz, threads.z));

    if (is_first && is_last) {
        CudaScopeTimerCallback cb(&timers.copy);

        cuda_copy_slice_x <<< blocks, threads >>> (data, params, 0, dx - 2);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Funtion cuda_copy_slice_x (forward) failed.\n");
            printf("Error: %s\n", cudaGetErrorString(err));
        }

        return;
    }

    MPI_Status comm_status;

    double *dev_buffer;
    cudaMalloc((void **) &dev_buffer, sizeof(double) * size);

    double send_buffer[size], recv_buffer[size];

    {
        CudaScopeTimerCallback cb(&timers.copy);

        cuda_buffer_slice_copy_x <<< blocks, threads >>> (dev_buffer, data, params, dx - 2, true);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Failed to fill send_buffer in send_recv_forward_x.\n");
            printf("Error: %s\n", cudaGetErrorString(err));
        }

        cudaMemcpy(send_buffer, dev_buffer, sizeof(double) * size, cudaMemcpyDeviceToHost);
    }

    {
        TimerScopeUnpauseCallback cb(timers.sendrecv);
        MPI_Sendrecv(send_buffer, size, MPI_DOUBLE, rank_next, 1,
                     recv_buffer, size, MPI_DOUBLE, rank_prev, 1,
                     comm_cart, &comm_status);
    }

    {
        CudaScopeTimerCallback cb(&timers.copy);

        cudaMemcpy(dev_buffer, recv_buffer, sizeof(double) * size, cudaMemcpyHostToDevice);

        cuda_buffer_slice_copy_x <<< blocks, threads >>> (dev_buffer, data, params, 0, false);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Failed to fill recv_buffer in send_recv_forward_x.\n");
            printf("Error: %s\n", cudaGetErrorString(err));
        }
    }

    cudaFree(dev_buffer);
}

void send_recv_backward_x(double *data, MPI_Comm& comm_cart, int rank_prev, int rank_next, bool is_first, bool is_last)
{
    cudaError_t err;

    const int dim1 = dy;
    const int dim2 = dz;
    const int size = dim1 * dim2;

    dim3 threads(1, 16, 16);
    dim3 blocks(1, split(dy, threads.y), split(dz, threads.z));

    if (is_first && is_last) {
        CudaScopeTimerCallback cb(&timers.copy);

        cuda_copy_slice_x <<< blocks, threads >>> (data, params, dx - 1, 2);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Funtion cuda_copy_slice_x (backward) failed.\n");
            printf("Error: %s\n", cudaGetErrorString(err));
        }

        return;
    }

    MPI_Status comm_status;

    double *dev_buffer;
    cudaMalloc((void **) &dev_buffer, sizeof(double) * size);

    double send_buffer[size], recv_buffer[size];

    {
        CudaScopeTimerCallback cb(&timers.copy);

        cuda_buffer_slice_copy_x <<< blocks, threads >>> (dev_buffer, data, params, (is_first) ? 2 : 1, true);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Failed to fill send_buffer in send_recv_backward_x.\n");
            printf("Error: %s\n", cudaGetErrorString(err));
        }

        cudaMemcpy(send_buffer, dev_buffer, sizeof(double) * size, cudaMemcpyDeviceToHost);
    }

    {
        TimerScopeUnpauseCallback cb(timers.sendrecv);
        MPI_Sendrecv(send_buffer, size, MPI_DOUBLE, rank_prev, 1,
                     recv_buffer, size, MPI_DOUBLE, rank_next, 1,
                     comm_cart, &comm_status);
    }

    {
        CudaScopeTimerCallback cb(&timers.copy);

        cudaMemcpy(dev_buffer, recv_buffer, sizeof(double) * size, cudaMemcpyHostToDevice);

        cuda_buffer_slice_copy_x <<< blocks, threads >>> (dev_buffer, data, params, dx - 1, false);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Failed to fill recv_buffer in send_recv_backward_x.\n");
            printf("Error: %s\n", cudaGetErrorString(err));
        }
    }

    cudaFree(dev_buffer);
}

void send_recv_forward_y(double *data, MPI_Comm& comm_cart, int rank_prev, int rank_next, bool is_first, bool is_last)
{
    cudaError_t err;

    const int dim1 = dx;
    const int dim2 = dz;
    const int size = dim1 * dim2;

    dim3 threads(16, 1, 16);
    dim3 blocks(split(dx, threads.x), 1, split(dz, threads.z));

    if (is_first && is_last) {
        CudaScopeTimerCallback cb(&timers.copy);

        cuda_copy_slice_y <<< blocks, threads >>> (data, params, 0, dy - 2);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Funtion cuda_copy_slice_y (forward) failed.\n");
            printf("Error: %s\n", cudaGetErrorString(err));
        }

        return;
    }

    MPI_Status comm_status;

    double *dev_buffer;
    cudaMalloc((void **) &dev_buffer, sizeof(double) * size);

    double send_buffer[size], recv_buffer[size];

    {
        CudaScopeTimerCallback cb(&timers.copy);

        cuda_buffer_slice_copy_y <<< blocks, threads >>> (dev_buffer, data, params, dy - 2, true);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Failed to fill send_buffer in send_recv_forward_y.\n");
            printf("Error: %s\n", cudaGetErrorString(err));
        }

        cudaMemcpy(send_buffer, dev_buffer, sizeof(double) * size, cudaMemcpyDeviceToHost);
    }

    {
        TimerScopeUnpauseCallback cb(timers.sendrecv);
        MPI_Sendrecv(send_buffer, size, MPI_DOUBLE, rank_next, 1,
                     recv_buffer, size, MPI_DOUBLE, rank_prev, 1,
                     comm_cart, &comm_status);
    }

    {
        CudaScopeTimerCallback cb(&timers.copy);

        cudaMemcpy(dev_buffer, recv_buffer, sizeof(double) * size, cudaMemcpyHostToDevice);

        cuda_buffer_slice_copy_y <<< blocks, threads >>> (dev_buffer, data, params, 0, false);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Failed to fill recv_buffer in send_recv_forward_y.\n");
            printf("Error: %s\n", cudaGetErrorString(err));
        }
    }

    cudaFree(dev_buffer);
}

void send_recv_backward_y(double *data, MPI_Comm& comm_cart, int rank_prev, int rank_next, bool is_first, bool is_last)
{
    cudaError_t err;

    const int dim1 = dx;
    const int dim2 = dz;
    const int size = dim1 * dim2;

    dim3 threads(16, 1, 16);
    dim3 blocks(split(dx, threads.x), 1, split(dz, threads.z));

    if (is_first && is_last) {
        CudaScopeTimerCallback cb(&timers.copy);

        cuda_copy_slice_y <<< blocks, threads >>> (data, params, dy - 1, 2);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Funtion cuda_copy_slice_y (backward) failed.\n");
            printf("Error: %s\n", cudaGetErrorString(err));
        }

        return;
    }

    MPI_Status comm_status;

    double *dev_buffer;
    cudaMalloc((void **) &dev_buffer, sizeof(double) * size);

    double send_buffer[size], recv_buffer[size];

    {
        CudaScopeTimerCallback cb(&timers.copy);

        cuda_buffer_slice_copy_y <<< blocks, threads >>> (dev_buffer, data, params, (is_first) ? 2 : 1, true);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Failed to fill send_buffer in send_recv_backward_y.\n");
            printf("Error: %s\n", cudaGetErrorString(err));
        }

        cudaMemcpy(send_buffer, dev_buffer, sizeof(double) * size, cudaMemcpyDeviceToHost);
    }

    {
        TimerScopeUnpauseCallback cb(timers.sendrecv);
        MPI_Sendrecv(send_buffer, size, MPI_DOUBLE, rank_prev, 1,
                     recv_buffer, size, MPI_DOUBLE, rank_next, 1,
                     comm_cart, &comm_status);
    }

    {
        CudaScopeTimerCallback cb(&timers.copy);

        cudaMemcpy(dev_buffer, recv_buffer, sizeof(double) * size, cudaMemcpyHostToDevice);

        cuda_buffer_slice_copy_y <<< blocks, threads >>> (dev_buffer, data, params, dy - 1, false);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Failed to fill recv_buffer in send_recv_backward_y.\n");
            printf("Error: %s\n", cudaGetErrorString(err));
        }
    }

    cudaFree(dev_buffer);
}

void send_recv_forward_z(double *data, MPI_Comm& comm_cart, int rank_prev, int rank_next, bool is_first, bool is_last)
{
    cudaError_t err;

    const int dim1 = dx;
    const int dim2 = dy;
    const int size = dim1 * dim2;

    dim3 threads(16, 16, 1);
    dim3 blocks(split(dx, threads.x), split(dy, threads.y), 1);

    if (is_first && is_last) {
        CudaScopeTimerCallback cb(&timers.copy);

        cuda_copy_slice_z <<< blocks, threads >>> (data, params, 0, dz - 2);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Funtion cuda_copy_slice_z (forward) failed.\n");
            printf("Error: %s\n", cudaGetErrorString(err));
        }

        return;
    }

    MPI_Status comm_status;

    double *dev_buffer;
    cudaMalloc((void **) &dev_buffer, sizeof(double) * size);

    double send_buffer[size], recv_buffer[size];

    {
        CudaScopeTimerCallback cb(&timers.copy);

        cuda_buffer_slice_copy_z <<< blocks, threads >>> (dev_buffer, data, params, dz - 2, true);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Failed to fill send_buffer in send_recv_forward_z.\n");
            printf("Error: %s\n", cudaGetErrorString(err));
        }

        cudaMemcpy(send_buffer, dev_buffer, sizeof(double) * size, cudaMemcpyDeviceToHost);
    }

    {
        TimerScopeUnpauseCallback cb(timers.sendrecv);
        MPI_Sendrecv(send_buffer, size, MPI_DOUBLE, rank_next, 1,
                     recv_buffer, size, MPI_DOUBLE, rank_prev, 1,
                     comm_cart, &comm_status);
    }

    {
        CudaScopeTimerCallback cb(&timers.copy);

        cudaMemcpy(dev_buffer, recv_buffer, sizeof(double) * size, cudaMemcpyHostToDevice);

        cuda_buffer_slice_copy_z <<< blocks, threads >>> (dev_buffer, data, params, 0, false);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Failed to fill recv_buffer in send_recv_forward_z.\n");
            printf("Error: %s\n", cudaGetErrorString(err));
        }
    }

    cudaFree(dev_buffer);
}

void send_recv_backward_z(double *data, MPI_Comm& comm_cart, int rank_prev, int rank_next, bool is_first, bool is_last)
{
    cudaError_t err;

    const int dim1 = dx;
    const int dim2 = dy;
    const int size = dim1 * dim2;

    dim3 threads(16, 16, 1);
    dim3 blocks(split(dx, threads.x), split(dy, threads.y), 1);

    if (is_first && is_last) {
        CudaScopeTimerCallback cb(&timers.copy);

        cuda_copy_slice_z <<< blocks, threads >>> (data, params, dz - 1, 2);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Funtion cuda_copy_slice_z (backward) failed.\n");
            printf("Error: %s\n", cudaGetErrorString(err));
        }

        return;
    }

    MPI_Status comm_status;

    double *dev_buffer;
    cudaMalloc((void **) &dev_buffer, sizeof(double) * size);

    double send_buffer[size], recv_buffer[size];

    {
        CudaScopeTimerCallback cb(&timers.copy);

        cuda_buffer_slice_copy_z <<< blocks, threads >>> (dev_buffer, data, params, (is_first) ? 2 : 1, true);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Failed to fill send_buffer in send_recv_backward_z.\n");
            printf("Error: %s\n", cudaGetErrorString(err));
        }

        cudaMemcpy(send_buffer, dev_buffer, sizeof(double) * size, cudaMemcpyDeviceToHost);
    }

    {
        TimerScopeUnpauseCallback cb(timers.sendrecv);
        MPI_Sendrecv(send_buffer, size, MPI_DOUBLE, rank_prev, 1,
                     recv_buffer, size, MPI_DOUBLE, rank_next, 1,
                     comm_cart, &comm_status);
    }

    {
        CudaScopeTimerCallback cb(&timers.copy);

        cudaMemcpy(dev_buffer, recv_buffer, sizeof(double) * size, cudaMemcpyHostToDevice);

        cuda_buffer_slice_copy_z <<< blocks, threads >>> (dev_buffer, data, params, dz - 1, false);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Failed to fill recv_buffer in send_recv_backward_z.\n");
            printf("Error: %s\n", cudaGetErrorString(err));
        }
    }

    cudaFree(dev_buffer);
}

#endif //PARALLEL_MPI_SEND_RECV_MPI_H
