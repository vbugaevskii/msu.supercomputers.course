#include <memory>
#include <cmath>

#include <iostream>
#include <fstream>
#include <sstream>

#include "process_split.h"
#include "border_conditions.h"
#include "send_recv_mpi.h"

TimersArray timers;

int rank = -1;

int N, K;
double hx, hy, hz, tau;
int dx, dy, dz;

const int LAYERS = 3;

inline double x(int i) { return i * hx; }
inline double y(int j) { return j * hy; }
inline double z(int k) { return k * hz; }

inline double phi_func(double x, double y, double z) {
    return sin(3 * x) * cos(2 * y) * sin(z);
}

inline double u_func(double x, double y, double z, double t) {
    return cos(sqrt(14) * t) * phi_func(x, y, z);
}

inline double f_func(double x, double y, double z, double t) {
    return 0;
}

/*
inline double phi_func(double x, double y, double z) {
    return sin(3 * x) * cos(2 * y) * sin(z);
}

inline double u_func(double x, double y, double z, double t) {
    return ( 1 + pow(t, 3.0) ) * phi_func(x, y, z);
}

inline double f_func(double x, double y, double z, double t) {
    return ( 6 * t + 14 * ( 1 + pow(t, 3.0) ) ) * phi_func(x, y, z);
}
 */

struct EstimateError {
    double mse;
    double max;

    EstimateError() : mse(0), max(0) {}
};

void estimate_error(EstimateError* p_error, const double *data, int t, int i_min, int j_min, int k_min) {
    double mse = 0;
    double max = 0;

    for (int p = 0; p < dx * dy * dz; p++) {
        int i = p % dx;
        int j = (p / dx) % dy;
        int k = (p / dx / dy) % dz;

        // пропускаем обменные области
        if (i == 0 || i == dx - 1)
            continue;

        if (j == 0 || j == dy - 1)
            continue;

        if (k == 0 || k == dz - 1)
            continue;

        double u_true = u_func(x(i_min + i - 1), y(j_min + j - 1), z(k_min + k - 1), t * tau);
        double u_pred = data[p];

        mse += pow(u_true - u_pred, 2);

        double u_abs = fabs(u_true - u_pred);
        if (u_abs > max)
            max = u_abs;
    }

    p_error->max = max;
    p_error->mse = mse;
}

int main(int argc, char **argv)
{
    timers.total.start();
    timers.init.start();

    N = (argc > 1) ? std::atoi(argv[1]) : 256;
    K = (argc > 2) ? std::atoi(argv[2]) : 20;

    const int nproc = (argc > 3) ? std::atoi(argv[3]) : 1;
    const bool is_compute = (argc > 4) ? std::atoi(argv[4]) : 0;

    hx = hy = hz = 16 * M_PI / (N - 1);
    tau = 0.004;

    // размерность декартовой решетки
    int ndim = 3;

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

    // инициализируем MPI и определяем ранг процесса
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // вывод информации о разбиении
    if (!rank) {
        std::cout << N << ' ' << K << ' ' << nproc << std::endl;
        for (int d = 0; d < ndim; d++) {
            std::cout << "axis" << d << '\t'
                      << dims[d] << '\t' << nodes[d] << std::endl;
        }
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

    // подсчет ошибки
    EstimateError error_cumm, error_curr, error_proc;

    // выделяем память
    double* u_data[LAYERS];
    for (int p = 0; p < LAYERS; p++)
        u_data[p] = new double[dx * dy * dz];

    timers.init.pause();

    // засекаем время
    MPITimer timer;
    timer.start();

    // заполняем для t = t0
    for (int p = 0; p < dx * dy * dz; p++) {
        int i = p % dx;
        int j = (p / dx) % dy;
        int k = (p / dx / dy) % dz;

        u_data[0][p] = phi_func(x(i_min + i - 1), y(j_min + j - 1), z(k_min + k - 1));
    }

    if (is_compute) {
        TimerScopePauseCallback callback(timer);

        error_curr.mse = 0;
        error_curr.max = 0;

        estimate_error(&error_proc, u_data[0], 0, i_min, j_min, k_min);

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
        for (int p = 0; p < dx * dy * dz; p++) {
            int i = p % dx;
            int j = (p / dx) % dy;
            int k = (p / dx / dy) % dz;

            // пропускаем обменные области
            if (i == 0 || i == dx - 1)
                continue;

            if (j == 0 || j == dy - 1)
                continue;

            if (k == 0 || k == dz - 1)
                continue;

            // пропускаем граничные области
            if (is_first[0] && i == 1 || (border_conditions[0] != 2) && is_last[0] && i == dx - 2)
                continue;

            if (is_first[1] && j == 1 || (border_conditions[1] != 2) && is_last[1] && j == dy - 2)
                continue;

            if (is_first[2] && k == 1 || (border_conditions[2] != 2) && is_last[2] && k == dz - 2)
                continue;

            if (n == 1) {
                // заполняем для t = t1;
                double f_value = f_func(x(i_min + i - 1), y(j_min + j - 1), z(k_min + k - 1), 0);

                u_data[n][p] = u_data[n-1][p] + 0.5 * tau * tau * (laplace(u_data[n-1], i, j, k) + f_value);
            } else {
                // заполняем для всех остальных t;
                double f_value = f_func(x(i_min + i - 1), y(j_min + j - 1), z(k_min + k - 1), (n - 1) * tau);

                u_data[n % LAYERS][p] = 2 * u_data[(n - 1) % LAYERS][p] - u_data[(n - 2) % LAYERS][p] \
                    + tau * tau * ( laplace(u_data[(n - 1) % LAYERS], i, j, k) + f_value );
            }
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

        {
            TimerScopeUnpauseCallback cb(timers.copy);

            switch (border_conditions[0]) {
                case 1:
                    border_condition_1st_x(u_data[n % LAYERS], is_first[0], is_last[0]);
                    break;
                case 2:
                    border_condition_2nd_x(u_data[n % LAYERS], is_first[0], is_last[0]);
                    break;
                default:;
            }

            switch (border_conditions[1]) {
                case 1:
                    border_condition_1st_y(u_data[n % LAYERS], is_first[1], is_last[1]);
                    break;
                case 2:
                    border_condition_2nd_y(u_data[n % LAYERS], is_first[1], is_last[1]);
                    break;
                default:;
            }

            switch (border_conditions[2]) {
                case 1:
                    border_condition_1st_z(u_data[n % LAYERS], is_first[2], is_last[2]);
                    break;
                case 2:
                    border_condition_2nd_z(u_data[n % LAYERS], is_first[2], is_last[2]);
                    break;
                default:;
            }
        }

        if (is_compute) {
            TimerScopePauseCallback callback(timer);

            error_curr.mse = 0;
            error_curr.max = 0;

            estimate_error(&error_proc, u_data[n % LAYERS], n, i_min, j_min, k_min);

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
        delete u_data[p];

    MPI_Finalize();

    timers.free.pause();
    timers.total.pause();

    if (!rank) {
        printf("\n");
        printf("Time total:     %.6f\n", timers.total.delta());
        printf("Time init:      %.6f\n", timers.init.delta());
        printf("Time logic:     %.6f\n", timer.delta());
        printf("Time sendrecv:  %.6f\n", timers.sendrecv.delta());
        printf("Time copy:      %.6f\n", timers.copy.delta());
        printf("Time free:      %.6f\n", timers.free.delta());
    }

    return 0;
}
