#ifndef PARALLEL_MPI_SEND_RECV_MPI_H
#define PARALLEL_MPI_SEND_RECV_MPI_H

#include "utils.h"

extern TimersArray timers;

void send_recv_forward_x(double *data, MPI_Comm& comm_cart, int rank_prev, int rank_next, bool is_first, bool is_last) {
    const int dim1 = dy;
    const int dim2 = dz;
    const int size = dim1 * dim2;

    if (is_first && is_last) {
        TimerScopeUnpauseCallback cb(timers.copy);

#pragma omp parallel for
        for (int p = 0; p < size; p++) {
            int j = p % dim1;
            int k = p / dim1;

            // положить в левую обменную область x_{N}
            data[index(0, j, k)] = data[index(dx - 2, j, k)];
        }
        return;
    }

    MPI_Status comm_status;

    double send_buffer[size], recv_buffer[size];

    {
        TimerScopeUnpauseCallback cb(timers.copy);

#pragma omp parallel for
        for (int p = 0; p < size; p++) {
            /*
             * Пересылаем:
             *  x_{N-1} для последнего процесса;
             *  x_{N} для всех остальных.
             */

            int j = p % dim1;
            int k = p / dim1;

            send_buffer[p] = data[index(dx - 2, j, k)];
        }
    }

    {
        TimerScopeUnpauseCallback cb(timers.sendrecv);

        MPI_Sendrecv(send_buffer, size, MPI_DOUBLE, rank_next, 1,
                     recv_buffer, size, MPI_DOUBLE, rank_prev, 1,
                     comm_cart, &comm_status);
    }

    {
        TimerScopeUnpauseCallback cb(timers.copy);

#pragma omp parallel for
        for (int p = 0; p < size; p++) {
            // копируем в обменную область
            int j = p % dim1;
            int k = p / dim1;

            data[index(0, j, k)] = recv_buffer[p];
        }
    }
}

void send_recv_backward_x(double *data, MPI_Comm& comm_cart, int rank_prev, int rank_next, bool is_first, bool is_last)
{
    const int dim1 = dy;
    const int dim2 = dz;
    const int size = dim1 * dim2;

    if (is_first && is_last) {
        TimerScopeUnpauseCallback cb(timers.copy);

#pragma omp parallel for
        for (int p = 0; p < size; p++) {
            int j = p % dim1;
            int k = p / dim1;

            // положить в правую обменную область x_{1}
            data[index(dx - 1, j, k)] = data[index(2, j,  k)];
        }
        return;
    }

    MPI_Status comm_status;

    double send_buffer[size], recv_buffer[size];

    {
        TimerScopeUnpauseCallback cb(timers.copy);

#pragma omp parallel for
        for (int p = 0; p < size; p++) {
            /*
             * Пересылаем:
             *  x_{1} для первого процесса;
             *  x_{0} для всех остальных.
             */

            int j = p % dim1;
            int k = p / dim1;

            send_buffer[p] = data[index((is_first) ? 2 : 1, j, k)];
        }
    }

    {
        TimerScopeUnpauseCallback cb(timers.sendrecv);

        MPI_Sendrecv(send_buffer, size, MPI_DOUBLE, rank_prev, 1,
                     recv_buffer, size, MPI_DOUBLE, rank_next, 1,
                     comm_cart, &comm_status);
    }

    {
        TimerScopeUnpauseCallback cb(timers.copy);

#pragma omp parallel for
        for (int p = 0; p < size; p++) {
            // копируем в обменную область
            int j = p % dim1;
            int k = p / dim1;

            data[index(dx - 1, j, k)] = recv_buffer[p];
        }
    }
}

void send_recv_forward_y(double *data, MPI_Comm& comm_cart, int rank_prev, int rank_next, bool is_first, bool is_last)
{
    const int dim1 = dx;
    const int dim2 = dz;
    const int size = dim1 * dim2;

    if (is_first && is_last) {
        TimerScopeUnpauseCallback cb(timers.copy);

#pragma omp parallel for
        for (int p = 0; p < size; p++) {
            int i = p % dim1;
            int k = p / dim1;

            // положить в левую обменную область x_{N}
            data[index(i, 0, k)] = data[index(i, dy - 2, k)];
        }
        return;
    }

    MPI_Status comm_status;

    double send_buffer[size], recv_buffer[size];

    {
        TimerScopeUnpauseCallback cb(timers.copy);

#pragma omp parallel for
        for (int p = 0; p < size; p++) {
            /*
             * Пересылаем:
             *  x_{N-1} для последнего процесса;
             *  x_{N} для всех остальных.
             */

            int i = p % dim1;
            int k = p / dim1;

            send_buffer[p] = data[index(i, dy - 2, k)];
        }
    }

    {
        TimerScopeUnpauseCallback cb(timers.sendrecv);

        MPI_Sendrecv(send_buffer, size, MPI_DOUBLE, rank_next, 1,
                     recv_buffer, size, MPI_DOUBLE, rank_prev, 1,
                     comm_cart, &comm_status);
    }

    {
        TimerScopeUnpauseCallback cb(timers.copy);

#pragma omp parallel for
        for (int p = 0; p < size; p++) {
            // копируем в обменную область
            int i = p % dim1;
            int k = p / dim1;

            data[index(i, 0, k)] = recv_buffer[p];
        }
    }
}

void send_recv_backward_y(double *data, MPI_Comm& comm_cart, int rank_prev, int rank_next, bool is_first, bool is_last)
{
    const int dim1 = dx;
    const int dim2 = dz;
    const int size = dim1 * dim2;

    if (is_first && is_last) {
        TimerScopeUnpauseCallback cb(timers.copy);

#pragma omp parallel for
        for (int p = 0; p < size; p++) {
            int i = p % dim1;
            int k = p / dim1;

            // положить в правую обменную область x_{1}
            data[index(i, dy - 1, k)] = data[index(i, 2, k)];
        }
        return;
    }

    MPI_Status comm_status;

    double send_buffer[size], recv_buffer[size];

    {
        TimerScopeUnpauseCallback cb(timers.copy);

#pragma omp parallel for
        for (int p = 0; p < size; p++) {
            /*
             * Пересылаем:
             *  x_{1} для первого процесса;
             *  x_{0} для всех остальных.
             */

            int i = p % dim1;
            int k = p / dim1;

            send_buffer[p] = data[index(i, (is_first) ? 2 : 1, k)];
        }
    }

    {
        TimerScopeUnpauseCallback cb(timers.sendrecv);

        MPI_Sendrecv(send_buffer, size, MPI_DOUBLE, rank_prev, 1,
                     recv_buffer, size, MPI_DOUBLE, rank_next, 1,
                     comm_cart, &comm_status);
    }

    {
        TimerScopeUnpauseCallback cb(timers.copy);

#pragma omp parallel for
        for (int p = 0; p < size; p++) {
            // копируем в обменную область
            int i = p % dim1;
            int k = p / dim1;

            data[index(i, dy - 1, k)] = recv_buffer[p];
        }
    }
}

void send_recv_forward_z(double *data, MPI_Comm& comm_cart, int rank_prev, int rank_next, bool is_first, bool is_last)
{
    const int dim1 = dx;
    const int dim2 = dy;
    const int size = dim1 * dim2;

    if (is_first && is_last) {
        TimerScopeUnpauseCallback cb(timers.copy);

#pragma omp parallel for
        for (int p = 0; p < size; p++) {
            int i = p % dim1;
            int j = p / dim1;

            // положить в левую обменную область x_{N}
            data[index(i, j, 0)] = data[index(i, j, dz - 2)];
        }
        return;
    }

    MPI_Status comm_status;

    double send_buffer[size], recv_buffer[size];

    {
        TimerScopeUnpauseCallback cb(timers.copy);

#pragma omp parallel for
        for (int p = 0; p < size; p++) {
            /*
             * Пересылаем:
             *  x_{N-1} для последнего процесса;
             *  x_{N} для всех остальных.
             */

            int i = p % dim1;
            int j = p / dim1;

            send_buffer[p] = data[index(i, j, dz - 2)];
        }
    }

    {
        TimerScopeUnpauseCallback cb(timers.sendrecv);

        MPI_Sendrecv(send_buffer, size, MPI_DOUBLE, rank_next, 1,
                     recv_buffer, size, MPI_DOUBLE, rank_prev, 1,
                     comm_cart, &comm_status);
    }

    {
        TimerScopeUnpauseCallback cb(timers.copy);

#pragma omp parallel for
        for (int p = 0; p < size; p++) {
            // копируем в обменную область
            int i = p % dim1;
            int j = p / dim1;

            data[index(i, j, 0)] = recv_buffer[p];
        }
    }
}

void send_recv_backward_z(double *data, MPI_Comm& comm_cart, int rank_prev, int rank_next, bool is_first, bool is_last)
{
    const int dim1 = dx;
    const int dim2 = dy;
    const int size = dim1 * dim2;

    if (is_first && is_last) {
        TimerScopeUnpauseCallback cb(timers.copy);

#pragma omp parallel for
        for (int p = 0; p < size; p++) {
            int i = p % dim1;
            int j = p / dim1;

            // положить в правую обменную область x_{1}
            data[index(i, j, dz - 1)] = data[index(i, j, 2)];
        }
        return;
    }

    MPI_Status comm_status;

    double send_buffer[size], recv_buffer[size];

    {
        TimerScopeUnpauseCallback cb(timers.copy);

#pragma omp parallel for
        for (int p = 0; p < size; p++) {
            /*
             * Пересылаем:
             *  x_{1} для первого процесса;
             *  x_{0} для всех остальных.
             */

            int i = p % dim1;
            int j = p / dim1;

            send_buffer[p] = data[index(i, j, (is_first) ? 2 : 1)];
        }
    }

    {
        TimerScopeUnpauseCallback cb(timers.sendrecv);

        MPI_Sendrecv(send_buffer, size, MPI_DOUBLE, rank_prev, 1,
                     recv_buffer, size, MPI_DOUBLE, rank_next, 1,
                     comm_cart, &comm_status);
    }

    {
        TimerScopeUnpauseCallback cb(timers.copy);

#pragma omp parallel for
        for (int p = 0; p < size; p++) {
            // копируем в обменную область
            int i = p % dim1;
            int j = p / dim1;

            data[index(i, j, dz - 1)] = recv_buffer[p];
        }
    }
}

#endif //PARALLEL_MPI_SEND_RECV_MPI_H
