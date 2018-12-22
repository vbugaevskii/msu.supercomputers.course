#ifndef PARALLEL_MPI_PROCESS_SPLIT_H
#define PARALLEL_MPI_PROCESS_SPLIT_H

#include <cmath>
#include <vector>
#include <algorithm>
#include <functional>

void find_prime_divisors(int n, std::vector<int> &result)
{
    // Eratosphen algorithm

    result.clear();

    std::vector<bool> is_removed(sqrt(n) + 1, false);

    for (int i = 2; i < is_removed.size(); i++) {
        if (is_removed[i])
            continue;

        for (int j = i; j < is_removed.size(); j += i)
            is_removed[i] = true;

        result.push_back(i);
    }
}

void find_divisors(int n, std::vector<int>& result)
{
    result.clear();

    std::vector<int> primes;
    find_prime_divisors(n, primes);

    for (int i = 0; i < primes.size(); i++) {
        int p = primes[i];

        while (n % p == 0) {
            result.push_back(p);
            n /= p;
        }

        if (n == 1)
            break;
    }

    if (n != 1)
        result.push_back(n);
}

void split_processes_by_grid(int *dims, int ndim, int nproc)
{
    std::vector<int> divisors;
    find_divisors(nproc, divisors);
    std::sort(divisors.begin(), divisors.end(), std::greater<int>());

    std::vector<int> split(ndim, 1);

    for (int i = 0; i < divisors.size(); i++) {
        *std::min_element(split.begin(), split.end()) *= divisors[i];
    }

    std::copy(split.begin(), split.end(), dims);
}

#endif //PARALLEL_MPI_PROCESS_SPLIT_H
