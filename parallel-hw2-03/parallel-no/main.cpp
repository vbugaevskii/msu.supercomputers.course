#include <memory>
#include <cmath>

#include <iostream>
#include <fstream>
#include <chrono>

int N = 128, K = 5;
double hx, hy, hz, tau;
int dx, dy, dz;

const int LAYERS = 3;

class Timer {
public:
    Timer() : t_delta(0) {}

    virtual void start() = 0;
    virtual void pause() = 0;
    virtual double delta() = 0;
    virtual void reset() = 0;

    virtual ~Timer() = default;

protected:
    double t_delta;
};

class ChronoTimer : public Timer {
public:
    ChronoTimer() : t_start(std::chrono::high_resolution_clock::now()), is_paused(true) {}

    void start() override {
        is_paused = false;
        t_start = std::chrono::high_resolution_clock::now();
    }

    void pause() override {
        if (!is_paused) {
            auto end = std::chrono::high_resolution_clock::now();
            double delta = std::chrono::duration_cast<std::chrono::microseconds>(end - t_start).count();
            t_delta += delta / 1e6;
        }
        is_paused = true;
    }

    double delta() override {
        double delta = 0;
        if (!is_paused) {
            auto end = std::chrono::high_resolution_clock::now();
            delta = std::chrono::duration_cast<std::chrono::microseconds>(end - t_start).count();
            delta /= 1e6;
        }
        return t_delta + delta;
    }

    void reset() override {
        t_delta = 0;
        start();
    }

private:
    std::chrono::high_resolution_clock::time_point t_start;
    bool is_paused;
};

class TimerScopePauseCallback {
public:
    explicit TimerScopePauseCallback(Timer& a_timer) : timer(a_timer) {
        timer.pause();
    }

    ~TimerScopePauseCallback() {
        timer.start();
    }

private:
    Timer& timer;
};

class TimerScopeUnpauseCallback {
public:
    explicit TimerScopeUnpauseCallback(Timer& a_timer) : timer(a_timer) {
        timer.start();
    }

    ~TimerScopeUnpauseCallback() {
        timer.pause();
    }

private:
    Timer& timer;
};

struct TimersArray {
    ChronoTimer total;
    ChronoTimer init;
    ChronoTimer free;
    ChronoTimer copy;
} timers;

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

struct EstimateError {
    double mse;
    double max;

    EstimateError() : mse(0), max(0) {}
};

void estimate_error(EstimateError* p_error, const double *data, int t) {
    double mse = 0;
    double max = 0;

    for (int p = 0; p < dx * dy * dz; p++) {
        int i = p % dx;
        int j = (p / dx) % dy;
        int k = (p / dx / dy) % dz;

        if (i == dx - 1 || j == dy - 1 || k == dz - 1)
            continue;

        double u_true = u_func(x(i), y(j), z(k), t * tau);
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

    N = (argc > 1) ? std::atoi(argv[1]) : 128;
    K = (argc > 2) ? std::atoi(argv[2]) : 20;

    const bool is_compute = (argc > 4) ? std::atoi(argv[4]) : 1;

    hx = hy = hz = 16 * M_PI / (N - 1);
    tau = 0.004;

    const int i_min = 0, j_min = 0, k_min = 0;
    dx = dy = dz = N + 1;

    // размерность декартовой решетки
    int ndim = 3;

    char border_conditions[ndim];
    border_conditions[0] = 1;
    border_conditions[1] = 2;
    border_conditions[2] = 1;

    // подсчет ошибки
    EstimateError error_cumm, error_curr;

    double* u_data[LAYERS];
    for (int p = 0; p < LAYERS; p++)
        u_data[p] = new double[dx * dy * dz];

    timers.init.pause();

    // засекаем время
    ChronoTimer timer;
    timer.start();

    // заполняем для t = t0
    for (int p = 0; p < dx * dy * dz; p++) {
        int i = p % dx;
        int j = (p / dx) % dy;
        int k = (p / dx / dy) % dz;

        u_data[0][p] = phi_func(x(i_min + i), y(j_min + j), z(k_min + k));
    }

    if (is_compute) {
        TimerScopePauseCallback callback(timer);

        estimate_error(&error_curr, u_data[0], 0);
        error_curr.mse /= pow(N, 3);
        error_cumm.mse += error_curr.mse;

        if (error_curr.max > error_cumm.max)
            error_cumm.max = error_curr.max;
    }

    printf("[iter %03d]", 0);
    if (is_compute)
        printf(" RMSE = %.6f; MAX = %.6f;", sqrt(error_curr.mse), error_curr.max);
    printf(" Time = %.6f sec.\n", timer.delta());

    // заполняем для остальных t
    for (int n = 1; n < K; n++) {
        for (int p = 0; p < dx * dy * dz; p++) {
            int i = p % dx;
            int j = (p / dx) % dy;
            int k = (p / dx / dy) % dz;

            // пропускаем обменные области
            if (i == dx - 1)
                continue;

            if (j == dy - 1)
                continue;

            if (k == dz - 1)
                continue;

            // пропускаем граничные области
            if (i == 0 || (border_conditions[0] != 2) && i == dx - 2)
                continue;

            if (j == 0 || (border_conditions[1] != 2) && j == dy - 2)
                continue;

            if (k == 0 || (border_conditions[2] != 2) && k == dz - 2)
                continue;

            if (n == 1) {
                // заполняем для t = t1;
                double f_value = f_func(x(i_min + i), y(j_min + j), z(k_min + k), 0);

                u_data[n][p] = u_data[n - 1][p] + 0.5 * tau * tau * (laplace(u_data[n - 1], i, j, k) + f_value);
            } else {
                // заполняем для всех остальных t;
                double f_value = f_func(x(i_min + i), y(j_min + j), z(k_min + k), (n - 1) * tau);

                u_data[n % LAYERS][p] = 2 * u_data[(n - 1) % LAYERS][p] - u_data[(n - 2) % LAYERS][p] \
                    + tau * tau * (laplace(u_data[(n - 1) % LAYERS], i, j, k) + f_value);
            }
        }

        {
            TimerScopeUnpauseCallback cb(timers.copy);

            if (border_conditions[0] == 1) {
                for (int k = 0; k < dz; k++) {
                    for (int j = 0; j < dy; j++) {
                        u_data[n % LAYERS][index(0, j, k)] = 0;
                        u_data[n % LAYERS][index(dx - 2, j, k)] = 0;
                    }
                }
            } else if (border_conditions[0] == 2) {
                for (int k = 0; k < dz; k++) {
                    for (int j = 0; j < dy; j++) {
                        u_data[n % LAYERS][index(0, j, k)] = u_data[n % LAYERS][index(dx - 2, j, k)];
                        u_data[n % LAYERS][index(dx - 1, j, k)] = u_data[n % LAYERS][index(1, j, k)];
                    }
                }
            }

            if (border_conditions[1] == 1) {
                for (int k = 0; k < dz; k++) {
                    for (int i = 0; i < dx; i++) {
                        u_data[n % LAYERS][index(i, 0, k)] = 0;
                        u_data[n % LAYERS][index(i, dy - 2, k)] = 0;
                    }
                }
            } else if (border_conditions[1] == 2) {
                for (int k = 0; k < dz; k++) {
                    for (int i = 0; i < dx; i++) {
                        u_data[n % LAYERS][index(i, 0, k)] = u_data[n % LAYERS][index(i, dy - 2, k)];
                        u_data[n % LAYERS][index(i, dy - 1, k)] = u_data[n % LAYERS][index(i, 1, k)];
                    }
                }
            }

            if (border_conditions[2] == 1) {
                for (int j = 0; j < dy; j++) {
                    for (int i = 0; i < dx; i++) {
                        u_data[n % LAYERS][index(i, j, 0)] = 0;
                        u_data[n % LAYERS][index(i, j, dz - 2)] = 0;
                    }
                }
            } else if (border_conditions[2] == 2) {
                for (int j = 0; j < dy; j++) {
                    for (int i = 0; i < dx; i++) {
                        u_data[n % LAYERS][index(i, j, 0)] = u_data[n % LAYERS][index(i, j, dz - 2)];
                        u_data[n % LAYERS][index(i, j, dz - 1)] = u_data[n % LAYERS][index(i, j, 1)];
                    }
                }
            }
        }

        if (is_compute) {
            TimerScopePauseCallback callback(timer);

            estimate_error(&error_curr, u_data[n % LAYERS], n);
            error_curr.mse /= pow(N, 3);
            error_cumm.mse += error_curr.mse;

            if (error_curr.max > error_cumm.max)
                error_cumm.max = error_curr.max;
        }

        printf("[iter %03d]", n);
        if (is_compute)
            printf(" RMSE = %.6f; MAX = %.6f;", sqrt(error_curr.mse), error_curr.max);
        printf(" Time = %.6f sec.\n", timer.delta());
    }

    timer.pause();

    if (is_compute)
        printf("Final RMSE = %.6f; MAX = %.6f\n", sqrt(error_cumm.mse / K), error_cumm.max);
    printf("Task elapsed in: %.6f sec.\n", timer.delta());

    timers.free.start();

    for (int p = 0; p < LAYERS; p++)
        delete u_data[p];

    timers.free.pause();
    timers.total.pause();

    printf("\n");
    printf("Time total:     %.6f\n", timers.total.delta());
    printf("Time init:      %.6f\n", timers.init.delta());
    printf("Time logic:     %.6f\n", timer.delta());
    printf("Time sendrecv:  NaN\n");
    printf("Time copy:      %.6f\n", timers.copy.delta());
    printf("Time free:      %.6f\n", timers.free.delta());

    return 0;
}
