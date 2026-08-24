#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>

using namespace std; 
// Macro for CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n", \
                    __func__, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void initialize(int* arr, int N, curandState *randStates)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= N)
        return;

    // Initialize the random number generator state
    curand_init(1234, tid, 0, &randStates[tid]);
    
    // Generate a uniform random number between 0 and 1, scale it, and cast to int
    arr[tid] = static_cast<int>(curand_uniform(&randStates[tid]) * 10);
}

__global__ void addArrVal(int* arr, int* sum, int N)
{
    __shared__ int local_sum;

    if (threadIdx.x == 0)
        local_sum = 0;

    // We have to wait for thread 0 to update the local_sum value to 0 before accumulating other values to it.
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N)
        atomicAdd(&local_sum, arr[tid]);

    // wait for other threads to add the value of arr[tid] to the local_sum variable.
    __syncthreads();

    if (threadIdx.x == 0)
        atomicAdd(sum, local_sum);
}

int main()
{
    int N = 1'000'000;
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    int* large_arr;
    int* sum;
    curandState *randStates;

    CUDA_CHECK(cudaMalloc(&large_arr, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&sum, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&randStates, N * sizeof(curandState)));

    CUDA_CHECK(cudaMemset(sum, 0, sizeof(int)));

    
    initialize<<<gridSize, blockSize>>>(large_arr, N, randStates);

    addArrVal<<<gridSize, blockSize>>>(large_arr, sum, N);

    CUDA_CHECK(cudaDeviceSynchronize());

    int result;
    CUDA_CHECK(cudaMemcpy(&result, sum, sizeof(int), cudaMemcpyDeviceToHost));

    cout << "Array Size = " << N << '\n';
    cout << "Sum of array = " << result << '\n';

    CUDA_CHECK(cudaFree(large_arr));
    CUDA_CHECK(cudaFree(sum));
    CUDA_CHECK(cudaFree(randStates));

    return 0;
}
