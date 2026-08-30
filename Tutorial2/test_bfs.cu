#include <iostream>
#include <vector>
#include <climits>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

// Macro to catch CUDA errors easily
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << endl; \
            exit(1); \
        } \
    } while(0)

struct DeviceCSRGraph {
    int* row_ptr;
    int* col_ind;
    double* values;
};

__global__ void dist_init(int* dist, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    dist[tid] = INT_MAX;
}

__device__ bool d_changed = false;

__global__ void BFS(int* dist, DeviceCSRGraph g, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= N || dist[tid] == INT_MAX) return;

    for (int i = g.row_ptr[tid]; i < g.row_ptr[tid + 1]; i++) {
        int neighbor = g.col_ind[i];
        int new_dist = dist[tid] + 1;
        
        // atomicMin returns the old value stored at the address
        int old_dist = atomicMin(&dist[neighbor], new_dist);
        
        if (new_dist < old_dist) {
            d_changed = true;
        }
    }
}

int main() {
    // Hardcoded Graph for Testing
    // Graph structure: 
    // 0 -> 1, 0 -> 2
    // 1 -> 3
    // 2 -> 3
    // 3 -> 4
    int nodes = 5;
    int edges = 5;
    int src = 0;
    int TPB = 256;
    
    // CSR Representation of the graph
    vector<int> h_row_ptr = {0, 2, 3, 4, 5, 5};
    vector<int> h_col_idx = {1, 2, 3, 3, 4};
    vector<double> h_values = {1.0, 1.0, 1.0, 1.0, 1.0}; // Values aren't used in BFS but kept for the struct

    int NUM_BLOCK = (nodes + TPB - 1) / TPB;

    vector<int> h_dist(nodes, 0);
    int* d_dist;
    CUDA_CHECK(cudaMalloc(&d_dist, sizeof(int) * nodes));
    
    dist_init<<<NUM_BLOCK, TPB>>>(d_dist, nodes);
    CUDA_CHECK(cudaDeviceSynchronize());

    int *d_row_ptr, *d_col_idx;
    double *d_values;
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (nodes + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values, edges * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_row_ptr, h_row_ptr.data(), (nodes + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, h_col_idx.data(), edges * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, h_values.data(), edges * sizeof(double), cudaMemcpyHostToDevice));

    DeviceCSRGraph d_G{d_row_ptr, d_col_idx, d_values};

    // Set source distance to 0
    int source_dist = 0;
    CUDA_CHECK(cudaMemcpy(d_dist + src, &source_dist, sizeof(int), cudaMemcpyHostToDevice));

    cout << "Running BFS from source node: " << src << "\n\n";

    bool h_changed = true;
    while (h_changed) {
        h_changed = false;
        CUDA_CHECK(cudaMemcpyToSymbol(d_changed, &h_changed, sizeof(bool)));
        
        BFS<<<NUM_BLOCK, TPB>>>(d_dist, d_G, nodes);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaMemcpyFromSymbol(&h_changed, d_changed, sizeof(bool)));
    }

    CUDA_CHECK(cudaMemcpy(h_dist.data(), d_dist, nodes * sizeof(int), cudaMemcpyDeviceToHost));

    for (int i = 0; i < h_dist.size(); i++) {
        if (h_dist[i] == INT_MAX) 
            cout << "Node : " << i << " -> UNREACHABLE\n";
        else 
            cout << "Node : " << i << " -> " << h_dist[i] << "\n";
    }

    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_col_idx));
    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_dist));

    return 0;
}