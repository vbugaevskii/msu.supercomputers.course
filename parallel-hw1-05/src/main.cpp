#include <iostream>
#include <cstdlib>
#include <memory>
#include <cmath>
#include <sstream>

#include "linalg.h"

#include "magma_v2.h"

static const double RAND_MAX_DOUBLE = RAND_MAX;

template <typename T>
int sgn(T val) {
    return ( T(0) < val ) - ( val < T(0) );
}

template <typename T>
std::string vector2string(const T *vector, int n) {
    std::stringstream ss;
    ss << '[';
    for (int i = 0; i < n; i++) {
        if (i > 0)
            ss << ", ";
        ss << vector[i];
    }
    ss << ']';
    return ss.str();
}

template <typename T>
std::string matrix2string(const T *matrix, int n_rows, int n_cols) {
    std::stringstream ss;
    ss << '[';
    for (int i = 0, k = 0; i < n_rows; i++, k += n_cols) {
        if (i > 0)
            ss << ',' << std::endl;
        ss << vector2string<T>(&matrix[k], n_cols);
    }
    ss << ']';
    return ss.str();
}

template <typename T>
std::string matrix2string(const T **matrix, int n_rows, int n_cols) {
    std::stringstream ss;
    ss << '[';
    for (int i = 0; i < n_rows; i++) {
        if (i > 0)
            ss << ',' << std::endl;
        ss << vector2string<T>(matrix[i], n_cols);
    }
    ss << ']';
    return ss.str();
}

void rnd_square_matrix(alglib::real_2d_array& matrix, int n, double min_val=0, double max_val=1) {
    std::unique_ptr<double> p_data(new double[n * n]);
    double *p_data_ = p_data.get();
    memset(p_data_, 0, n * n * sizeof(double));

    for (int i = 0; i < n; i++) {
        double *p_curr = &p_data_[i * (n + 1)];
        *p_curr = (rand() / RAND_MAX_DOUBLE) * (max_val - min_val) + min_val;
        // Struggle with redundant zeros
        if ( fabs(*p_curr) < 1e-3 ) {
            *p_curr += sgn<double>( p_data_[i * (n + 1)] ) * 1e-3;
        }
    }

    // std::cout << "[matrix A]" << std::endl;
    // std::cout << matrix2string<double>(p_data_, n, n) << '\n' << std::endl;

    matrix.setcontent(n, n, p_data.get());
    alglib::smatrixrndmultiply(matrix, n);
}

void matrix_A_init(float *p_matrix, int n, double min_val=-5, double max_val=5) {
    alglib::real_2d_array matrix;
    rnd_square_matrix(matrix, n, min_val, max_val);

    for (int i = 0, k = 0; i < matrix.rows(); i++) {
        for (int j = 0; j < matrix.cols(); j++, k++) {
            p_matrix[k] = static_cast<float>( matrix(i, j) );
        }
    }
}

void vector_y_init(float *p_vector, int n, double min_val=-5, double max_val=5) {
    for (int i = 0; i < n; i++) {
        p_vector[i] = (rand() / RAND_MAX_DOUBLE) * (max_val - min_val) + min_val;
    }
}

double measure_residual(const float *matrix_A, const float *vector_x, const float* vector_y, int n) {
    std::unique_ptr<double> p_data(new double[n * n]);
    double *p_data_ = p_data.get();

    alglib::real_2d_array matrix_A_;
    for (int k = 0; k < n * n; k++)
        p_data_[k] = static_cast<double>(matrix_A[k]);
    matrix_A_.setcontent(n, n, p_data_);

    alglib::real_2d_array vector_x_;
    for (int k = 0; k < n; k++)
        p_data_[k] = static_cast<double>(vector_x[k]);
    vector_x_.setcontent(n, 1, p_data_);

    alglib::real_2d_array vector_y_;
    memset(p_data_, 0, n * sizeof(double));
    vector_y_.setcontent(n, 1, p_data_);

    alglib::rmatrixgemm(n, 1, n, 1, matrix_A_, 0, 0, 0, vector_x_, 0, 0, 0, 0, vector_y_, 0, 0);

    // std::cout << "A matrix:\n" << matrix_A_.tostring(5) << std::endl;
    // std::cout << "x vector:\n" << vector_x_.tostring(5) << std::endl;
    // std::cout << "y vector:\n" << vector_y_.tostring(5) << std::endl;

    double residiual = 0;
    for (int k = 0; k < n; k++)
        residiual += pow(vector_y[k] - vector_y_(k, 0), 2);
    residiual = sqrt(residiual);

    return residiual;
}

class ScopeMagmaTimer {
public:
    ScopeMagmaTimer(const std::string& msg, real_Double_t *p_elapsed=nullptr) :
        msg_(msg),
        p_elapsed_(p_elapsed)
    {
        start_ = magma_sync_wtime(NULL);
    }

    ~ScopeMagmaTimer() {
        real_Double_t end_ = magma_sync_wtime(NULL);
        real_Double_t elapsed = end_ - start_;

        char buffer[32];
        sprintf(buffer, "%7.5f sec.", elapsed);
        std::cout << msg_ << " elapsed in " << buffer << std::endl;

        if (p_elapsed_)
            *p_elapsed_ = elapsed;
    }
private:
    real_Double_t start_;
    real_Double_t *p_elapsed_;
    std::string msg_;
};

void run(const int N, const bool valid) {
    float *matrix_A, *vector_y, *vector_x;
    magmaFloat_ptr matrix_A_gpu, vector_y_gpu;

    magma_init();

    magma_queue_t queue = NULL;
    magma_int_t dev = 0;
    magma_queue_create(dev, &queue);

    magma_int_t err;

    // Reserve space for matrix_A to CPU
    err = magma_smalloc_cpu(&matrix_A, N * N);
    err = magma_smalloc(&matrix_A_gpu, N * N);

    // Reserve space for vector_y (right hand side) to CPU
    err = magma_smalloc_cpu(&vector_y, N);
    err = magma_smalloc(&vector_y_gpu, N);

    // Reserve space for vector_x
    err = magma_smalloc_cpu(&vector_x, N);

    magma_int_t *piv, info;
    piv = static_cast<magma_int_t *>( calloc(N * 2, sizeof(magma_int_t)) );

    ScopeMagmaTimer timer("total");

    // Init matrix_A and vector_y
    {
    ScopeMagmaTimer timer("matrix init");

    matrix_A_init(matrix_A, N, -5, 5);
    vector_y_init(vector_y, N, -5, 5);
    }

    // Load matrix_A and vector_y to GPU
    {
    ScopeMagmaTimer timer("matrix copy (CPU -> GPU)");

    magma_ssetmatrix(N, N, matrix_A, N, matrix_A_gpu, N, queue);
    magma_ssetmatrix(N, 1, vector_y, N, vector_y_gpu, N, queue);
    }

    {
    ScopeMagmaTimer timer("magma_sgetrf_gpu");
    magma_sgetrf_gpu(N, N, matrix_A_gpu, N, piv, &info);
    }

    {
    ScopeMagmaTimer timer("magma_sgetrs_gpu");
    magma_sgetrs_gpu(MagmaNoTrans, N, 1, matrix_A_gpu, N, piv, vector_y_gpu, N, &info);
    }

    // Load vector_x from GPU to CPU
    {
    ScopeMagmaTimer timer("matrix copy (GPU -> CPU)");
    magma_sgetmatrix(N, 1, vector_y_gpu, N, vector_x, N, queue);
    }

    /*
    std::cout << "[matrix A]" << std::endl;
    std::cout << matrix2string<float>(matrix_A, N, N) << '\n' << std::endl;

    std::cout << "[vector y]" << std::endl;
    std::cout << vector2string<float>(vector_y, N) << '\n' << std::endl;

    std::cout << "[vector x]" << std::endl;
    std::cout << vector2string<float>(vector_x, N) << '\n' << std::endl;
    */

    if (valid) {
        ScopeMagmaTimer timer("check operation");

        char buffer[32];
        double res = measure_residual(matrix_A, vector_x, vector_y, N);
        sprintf(buffer, "%.5f", res);
        std::cout << "Residiual: " << buffer << std::endl;
    }

    magma_free(matrix_A_gpu);
    magma_free_cpu(matrix_A);

    magma_free(vector_y_gpu);
    magma_free_cpu(vector_y);

    magma_free_cpu(vector_x);

    free(piv);

    magma_queue_destroy(queue);
    magma_finalize();
}

int main(int argc, char **argv) {
    const int N = std::atoi(argv[1]); // N = 32;
    const bool valid = (argc > 2) ? std::atoi(argv[2]) : false;

    std::cout << "matrix size: " << N << std::endl;
    std::cout << "valid: " << valid << std::endl;
    std::cout << std::endl;

    run(N, valid);

    return 0;
}
