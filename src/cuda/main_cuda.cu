#include <iostream>
#include <cuda_runtime.h>

__global__ void dummy_kernel(int *x) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx == 0) x[0] = 42;
}

int main(int argc, char** argv) {
    // ensure CUDA runtime is initialized
    cudaError_t err = cudaFree(0);
    if (err != cudaSuccess) {
        std::cerr << "CUDA init failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    int *d;
    cudaMalloc(&d, sizeof(int));
    dummy_kernel<<<1, 32>>>(d);
    cudaDeviceSynchronize();

    int h = 0;
    cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Hello CUDA: kernel wrote: " << h << std::endl;

    cudaFree(d);
    return 0;
}
