#include <cstdio>
#include <iostream>

#include <cmath>
#include <memory>

#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#include "process_split.h"
#include "border_conditions.h"
#include "send_recv_mpi.h"

TimersArray timers;

int rank = -1;

int N, K;
int dx, dy, dz;
double hx, hy, hz, tau;

const int LAYERS = 3;

cell_info_t params;

__host__ __device__ double phi_func(double x, double y, double z) {
    return sin(3 * x) * cos(2 * y) * sin(z);
}

__host__ __device__ double u_func(double x, double y, double z, double t) {
    return cos(sqrt(14.0) * t) * phi_func(x, y, z);
}

__host__ __device__ double f_func(double x, double y, double z, double t) {
    return 0;
}

/*
__host__ __device__ double phi_func(double x, double y, double z) {
    return sin(3 * x) * cos(2 * y) * sin(z);
}

__host__ __device__ double u_func(double x, double y, double z, double t) {
    return ( 1 + pow(t, 3.0) ) * phi_func(x, y, z);
}

__host__ __device__ double f_func(double x, double y, double z, double t) {
    return ( 6 * t + 14 * ( 1 + pow(t, 3.0) ) ) * phi_func(x, y, z);
}
 */

struct EstimateError {
    double mse;
    double max;

    EstimateError() : mse(0), max(0) {}
};

__global__ void cuda_task_init(double *data, cell_info_t params)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;

    if (i >= params.dx || j >= params.dy || k >= params.dz)
        return;

    double x = (params.i_min + i - 1) * params.hx;
    double y = (params.j_min + j - 1) * params.hy;
    double z = (params.k_min + k - 1) * params.hz;

    data[index(i, j, k, params)] = phi_func(x, y, z);
}

__global__ void cuda_task_iter(double *p_next, double *p_curr, double *p_prev, int n, cell_info_t params)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x + 1;
    int j = blockDim.y * blockIdx.y + threadIdx.y + 1;
    int k = blockDim.z * blockIdx.z + threadIdx.z + 1;

    // пропускаем правую обменную область и выход за границы
    if (i >= params.dx - 1 || j >= params.dy - 1 || k >= params.dz - 1)
        return;

    bool is_first[ndim], is_last[ndim];
    unpack_fl_mask(is_first, is_last, params.fl_mask);

    char border_conditions[ndim];
    unpack_bc_mask(border_conditions, params.bc_mask);

    // пропускаем граничные области
    if (is_first[0] && i == 1 || (border_conditions[0] != 2) && is_last[0] && i == params.dx - 2)
        return;

    if (is_first[1] && j == 1 || (border_conditions[1] != 2) && is_last[1] && j == params.dy - 2)
        return;

    if (is_first[2] && k == 1 || (border_conditions[2] != 2) && is_last[2] && k == params.dz - 2)
        return;

    int p = index(i, j, k, params);

    double x = (params.i_min + i - 1) * params.hx;
    double y = (params.j_min + j - 1) * params.hy;
    double z = (params.k_min + k - 1) * params.hz;

    if (n == 1) {
        // заполняем для t = t1;
        double f_value = f_func(x, y, z, 0);

        p_next[p] = p_curr[p] + 0.5 * params.tau * params.tau * \
            ( laplace(p_curr, i, j, k, params) + f_value );
    } else {
        // заполняем для всех остальных t;
        double f_value = f_func(x, y, z, (n - 1) * params.tau);

        p_next[p] = 2 * p_curr[p] - p_prev[p] + params.tau * params.tau * \
            ( laplace(p_curr, i, j, k, params) + f_value );
    }
}

__global__ void cuda_mse_error(double *err, const double *data, cell_info_t params, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x + 1;
    int j = blockDim.y * blockIdx.y + threadIdx.y + 1;
    int k = blockDim.z * blockIdx.z + threadIdx.z + 1;

    // пропускаем правую обменную область и выход за границы
    if (i >= params.dx - 1 || j >= params.dy - 1 || k >= params.dz - 1)
        return;

    int p = (i - 1) + (j - 1) * (params.dx - 2) + (k - 1) * (params.dx - 2) * (params.dy - 2);

    double x = (params.i_min + i - 1) * params.hx;
    double y = (params.j_min + j - 1) * params.hy;
    double z = (params.k_min + k - 1) * params.hz;

    double u_true = u_func(x, y, z, n * params.tau);
    err[p] = pow(data[index(i, j, k, params)] - u_true, 2.0);
}

__global__ void cuda_max_error(double *err, const double *data, cell_info_t params, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x + 1;
    int j = blockDim.y * blockIdx.y + threadIdx.y + 1;
    int k = blockDim.z * blockIdx.z + threadIdx.z + 1;

    // пропускаем правую обменную область и выход за границы
    if (i >= params.dx - 1 || j >= params.dy - 1 || k >= params.dz - 1)
        return;

    int p = (i - 1) + (j - 1) * (params.dx - 2) + (k - 1) * (params.dx - 2) * (params.dy - 2);

    double x = (params.i_min + i - 1) * params.hx;
    double y = (params.j_min + j - 1) * params.hy;
    double z = (params.k_min + k - 1) * params.hz;

    double u_true = u_func(x, y, z, n * params.tau);
    double value = data[index(i, j, k, params)] - u_true;
    err[p] = (value < 0) ? -value : value;
}

int main(int argc, char **argv)
{
    // инициализируем MPI и определяем ранг процесса
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    timers.total.start();
    timers.init.start();

    N = (argc > 1) ? std::atoi(argv[1]) : 256;
    K = (argc > 2) ? std::atoi(argv[2]) : 20;

    const int nproc = (argc > 3) ? std::atoi(argv[3]) : 1;
    const bool is_compute = (argc > 4) ? std::atoi(argv[4]) : 0;

    hx = hy = hz = 16 * M_PI / (N - 1);
    tau = 0.004;

    char border_conditions[ndim];
    border_conditions[0] = 1;
    border_conditions[1] = 2;
    border_conditions[2] = 1;

    // число процессов по каждой из оси решетки
    int dims[ndim];
    split_processes_by_grid(dims, ndim, nproc);

    // решетка является периодической (для установки граничных условий)
    int periods[ndim];
    for (int d = 0; d < ndim; d++)
        periods[d] = 1;

    // число узлов решетки для процесса по каждой из осей
    int nodes[ndim];
    for (int d = 0; d < ndim; d++) {
        nodes[d] = (int) ceil(N / (double) dims[d]);
        if (!nodes[d]) {
            std::cerr << "[ERROR] Invalid grid split" << std::endl;
            return 1;
        }
    }

    // равномерно распределяем процессы между GPU
    int num_cuda_devices;
    cudaGetDeviceCount(&num_cuda_devices);
    cudaSetDevice(rank % num_cuda_devices);

    // int curr_cuda_device;
    // cudaGetDevice(&curr_cuda_device);
    // std::cout << rank << ' ' << curr_cuda_device << ' ' << std::endl;

    // вывод информации о разбиении
    if (!rank) {
        std::cout << N << ' ' << K << ' ' << nproc << std::endl;
        for (int d = 0; d < ndim; d++) {
            std::cout << "axis" << d << '\t'
                      << dims[d] << '\t' << nodes[d] << std::endl;
        }
        std::cout << "Number of cuda devices: " << num_cuda_devices << std::endl;
    }

    // создаем топологию
    MPI_Comm comm_cart;
    MPI_Cart_create(MPI_COMM_WORLD, ndim, dims, periods, 0, &comm_cart);

    // координаты процесса в системе декартовой решетки
    int coords[ndim];
    MPI_Cart_coords(comm_cart, rank, ndim, coords);

    // вычисляем соседей для процесса по каждой из осей
    int rank_prev[ndim], rank_next[ndim];
    for (int d = 0; d < ndim; d++) {
        MPI_Cart_shift(comm_cart, d, +1, &rank_prev[d], &rank_next[d]);
    }

    // индикаторы того, что процесс является первым и/или последним по каждой из осей
    bool is_first[ndim], is_last[ndim];
    for (int d = 0; d < ndim; d++) {
        is_first[d] = (!coords[d]);
        is_last[d] = (coords[d] == dims[d] - 1);
    }

    // минимальные и максимальные рабочие индексы
    const int i_min = coords[0] * nodes[0], i_max = std::min(N, (coords[0] + 1) * nodes[0]) - 1;
    const int j_min = coords[1] * nodes[1], j_max = std::min(N, (coords[1] + 1) * nodes[1]) - 1;
    const int k_min = coords[2] * nodes[2], k_max = std::min(N, (coords[2] + 1) * nodes[2]) - 1;

    // ширина области в индексах
    // храним еще и обменные области (по 2е на каждую ось), помимо рабочих областей
    dx = i_max - i_min + 1 + 2;
    dy = j_max - j_min + 1 + 2;
    dz = k_max - k_min + 1 + 2;

    params.dx = dx;
    params.dy = dy;
    params.dz = dz;
    params.hx = hx;
    params.hy = hy;
    params.hz = hz;
    params.tau = tau;
    params.i_min = i_min;
    params.j_min = j_min;
    params.k_min = k_min;
    params.fl_mask = pack_fl_mask(is_first, is_last);
    params.bc_mask = pack_bc_mask(border_conditions);

    // подсчет ошибки
    EstimateError error_cumm, error_curr, error_proc;

    cudaError_t err;

    // выделяем память на GPU
    double *u_data[LAYERS], *u_error;
    for (int p = 0; p < LAYERS; p++)
        cudaMalloc((void **) &u_data[p], sizeof(double) * dx * dy * dz);
    cudaMalloc((void **) &u_error, sizeof(double) * (dx - 2) * (dy - 2) * (dz - 2));

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Memory GPU allocation failed.\n");
        printf("Error: %s\n", cudaGetErrorString(err));
    }

    timers.init.pause();

    // засекаем время
    MPITimer timer;
    timer.start();

    // определяем разбиение на GPU (обменные области заполняются, но не вычисляются)
    dim3 threads(8, 8, 8);
    dim3 blocks(split(dx - 2, threads.x), split(dy - 2, threads.y), split(dz - 2, threads.z));

    // заполняем для t = t0
    {
    dim3 blocks(split(dx, threads.x), split(dy, threads.y), split(dz, threads.z));
    cuda_task_init <<< blocks, threads >>> (u_data[0], params);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Function cuda_task_init has failed.\n");
        printf("Error: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();

    if (is_compute) {
        TimerScopePauseCallback callback(timer);

        error_curr.mse = 0;
        error_curr.max = 0;

        cuda_mse_error <<< blocks, threads >>> (u_error, u_data[0], params, 0);
        cudaDeviceSynchronize();
        error_proc.mse = thrust::reduce(
            thrust::device,
            u_error, u_error + (dx - 2) * (dy - 2) * (dz - 2),
            0.0, thrust::plus<double>()
        );

        cuda_max_error <<< blocks, threads >>> (u_error, u_data[0], params, 0);
        cudaDeviceSynchronize();
        error_proc.max = thrust::reduce(
            thrust::device,
            u_error, u_error + (dx - 2) * (dy - 2) * (dz - 2),
            0.0, thrust::maximum<double>()
        );

        MPI_Reduce(&error_proc.mse, &error_curr.mse, 1, MPI_DOUBLE, MPI_SUM, 0, comm_cart);
        MPI_Reduce(&error_proc.max, &error_curr.max, 1, MPI_DOUBLE, MPI_MAX, 0, comm_cart);

        if (!rank) {
            error_curr.mse /= pow(N, 3);
            error_cumm.mse += error_curr.mse;

            if (error_curr.max > error_cumm.max)
                error_cumm.max = error_curr.max;
        }
    }

    if (!rank) {
        printf("[iter %03d]", 0);
        if (is_compute)
            printf(" RMSE = %.6f; MAX = %.6f;", sqrt(error_curr.mse), error_curr.max);
        printf(" Time = %.6f sec.\n", timer.delta());
    }

    // заполняем для остальных t
    for (int n = 1; n < K; n++) {
        cuda_task_iter <<< blocks, threads >>> (
            u_data[n % LAYERS], u_data[(n - 1) % LAYERS], u_data[(n - 2) % LAYERS], n, params);
        cudaDeviceSynchronize();

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Function cuda_task_iter has failed. Current iteration: %d\n", n);
            printf("Error: %s\n", cudaGetErrorString(err));
        }

        // обмены граничными областями между процессами по оси X
        send_recv_forward_x (u_data[n % LAYERS], comm_cart, rank_prev[0], rank_next[0], is_first[0], is_last[0]);
        send_recv_backward_x(u_data[n % LAYERS], comm_cart, rank_prev[0], rank_next[0], is_first[0], is_last[0]);

        // обмены граничными областями между процессами по оси Y
        send_recv_forward_y (u_data[n % LAYERS], comm_cart, rank_prev[1], rank_next[1], is_first[1], is_last[1]);
        send_recv_backward_y(u_data[n % LAYERS], comm_cart, rank_prev[1], rank_next[1], is_first[1], is_last[1]);

        // обмены граничными областями между процессами по оси Z
        send_recv_forward_z (u_data[n % LAYERS], comm_cart, rank_prev[2], rank_next[2], is_first[2], is_last[2]);
        send_recv_backward_z(u_data[n % LAYERS], comm_cart, rank_prev[2], rank_next[2], is_first[2], is_last[2]);

        cudaDeviceSynchronize();

        {
            CudaScopeTimerCallback cb(&timers.copy);

            switch (border_conditions[0]) {
                case 1: border_condition_1st_x(u_data[n % LAYERS], is_first[0], is_last[0]); break;
                case 2: border_condition_2nd_x(u_data[n % LAYERS], is_first[0], is_last[0]); break;
                default: ;
            }

            switch (border_conditions[1]) {
                case 1: border_condition_1st_y(u_data[n % LAYERS], is_first[1], is_last[1]); break;
                case 2: border_condition_2nd_y(u_data[n % LAYERS], is_first[1], is_last[1]); break;
                default: ;
            }

            switch (border_conditions[2]) {
                case 1: border_condition_1st_z(u_data[n % LAYERS], is_first[2], is_last[2]); break;
                case 2: border_condition_2nd_z(u_data[n % LAYERS], is_first[2], is_last[2]); break;
                default: ;
            }
        }

        cudaDeviceSynchronize();

        if (is_compute) {
            TimerScopePauseCallback callback(timer);

            error_curr.mse = 0;
            error_curr.max = 0;

            cuda_mse_error <<< blocks, threads >>> (u_error, u_data[n % LAYERS], params, n);
            cudaDeviceSynchronize();
            error_proc.mse = thrust::reduce(
                thrust::device,
                u_error, u_error + (dx - 2) * (dy - 2) * (dz - 2),
                0.0, thrust::plus<double>()
            );

            cuda_max_error <<< blocks, threads >>> (u_error, u_data[n % LAYERS], params, n);
            cudaDeviceSynchronize();
            error_proc.max = thrust::reduce(
                thrust::device,
                u_error, u_error + (dx - 2) * (dy - 2) * (dz - 2),
                0.0, thrust::maximum<double>()
            );

            MPI_Reduce(&error_proc.mse, &error_curr.mse, 1, MPI_DOUBLE, MPI_SUM, 0, comm_cart);
            MPI_Reduce(&error_proc.max, &error_curr.max, 1, MPI_DOUBLE, MPI_MAX, 0, comm_cart);

            if (!rank) {
                error_curr.mse /= pow(N, 3);
                error_cumm.mse += error_curr.mse;

                if (error_curr.max > error_cumm.max)
                    error_cumm.max = error_curr.max;
            }
        }

        if (!rank) {
            printf("[iter %03d]", n);
            if (is_compute)
                printf(" RMSE = %.6f; MAX = %.6f;", sqrt(error_curr.mse), error_curr.max);
            printf(" Time = %.6f sec.\n", timer.delta());
        }
    }

    timer.pause();

    if (!rank) {
        if (is_compute)
            printf("Final RMSE = %.6f; MAX = %.6f\n", sqrt(error_cumm.mse / K), error_cumm.max);
        printf("Task elapsed in: %.6f sec.\n", timer.delta());
    }

    timers.free.start();

    // освобождаем память
    for (int p = 0; p < LAYERS; p++)
        cudaFree(u_data[p]);
    cudaFree(u_error);

    timers.free.pause();
    timers.total.pause();

    MPI_Finalize();

    if (!rank) {
        printf("\n");
        printf("Time total:     %.6f\n", timers.total.delta());
        printf("Time init:      %.6f\n", timers.init.delta());
        printf("Time logic:     %.6f\n", timer.delta());
        printf("Time sendrecv:  %.6f\n", timers.sendrecv.delta());
        printf("Time copy:      %.6f\n", timers.copy);
        printf("Time free:      %.6f\n", timers.free.delta());
    }

    return 0;
}
