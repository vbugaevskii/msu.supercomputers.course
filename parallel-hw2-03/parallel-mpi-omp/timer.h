#ifndef PARALLEL_MPI_TIMER_H
#define PARALLEL_MPI_TIMER_H

#include <chrono>
#include <mpi.h>

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

class OMPTimer : public Timer {
public:
    OMPTimer() : t_start(0.0), is_paused(true) {}

    void start() override {
        is_paused = false;
        t_start = omp_get_wtime();
    }

    void pause() override {
        if (!is_paused)
            t_delta += omp_get_wtime() - t_start;
        is_paused = true;
    }

    double delta() override {
        double delta = 0;
        if (!is_paused)
            delta = omp_get_wtime() - t_start;
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

class MPITimer : public Timer {
public:
    MPITimer() : t_start(0.0), is_paused(true) {}

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
            delta = MPI_Wtime() - t_start;
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
    ChronoTimer() : t_start(), is_paused(true) {}

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

#endif //PARALLEL_MPI_TIMER_H
