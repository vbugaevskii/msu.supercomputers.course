#ifndef PARALLEL_MPI_TIMER_H
#define PARALLEL_MPI_TIMER_H

#include <mpi.h>
#include <ctime>

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

class MPITimer : public Timer {
public:
    MPITimer() : t_start(MPI_Wtime()), is_paused(true) {}

    void start() override {
        is_paused = false;
        t_start = MPI_Wtime();
    }

    void pause() override {
        if (!is_paused)
            t_delta += MPI_Wtime() - t_start;
        is_paused = true;
    }

    double delta() override {
        double delta = 0;
        if (!is_paused)
            delta =  (MPI_Wtime() - t_start);
        return t_delta + delta;
    }

    void reset() override {
        t_delta = 0;
        start();
    }

private:
    double t_start;
    bool is_paused;
};

class ChronoTimer : public Timer {
public:
    ChronoTimer() : t_start(clock()), is_paused(true) {}

    void start() override {
        is_paused = false;
        t_start = clock();
    }

    void pause() override {
        if (!is_paused)
            t_delta += ((double) (clock() - t_start)) / CLOCKS_PER_SEC;
        is_paused = true;
    }

    double delta() override {
        double delta = 0;
        if (!is_paused)
            delta = ((double) (clock() - t_start)) / CLOCKS_PER_SEC;
        return t_delta + delta;
    }

    void reset() override {
        t_delta = 0;
        start();
    }

private:
    clock_t t_start;
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

class CudaScopeTimerCallback
{
public:
    CudaScopeTimerCallback(double *p_delta) : p_delta(p_delta) {
        cudaEventCreate(&t_start);
        cudaEventCreate(&t_stop);

        cudaEventRecord(t_start, 0);
    }

    ~CudaScopeTimerCallback() {
        cudaEventRecord(t_stop, 0);
        cudaEventSynchronize(t_stop);

        float t_elapsed = 0;
        cudaEventElapsedTime(&t_elapsed, t_start, t_stop);

        *p_delta += t_elapsed / 1000.0;

        cudaEventDestroy(t_stop);
        cudaEventDestroy(t_start);
    }

private:
    double *p_delta;
    cudaEvent_t t_start, t_stop;
};

#endif //PARALLEL_MPI_TIMER_H
