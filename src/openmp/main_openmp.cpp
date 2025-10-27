#include <iostream>
#include <omp.h>

int main() {
    int nthreads = 0;
    #pragma omp parallel
    {
        #pragma omp atomic
        nthreads++;
    }

    std::cout << "Hello from OpenMP! Threads used: " << nthreads << std::endl;
    return 0;
}
