#include<bits/stdc++.h>
#include "CSRGraph.hpp"
#include "../utils/utils.hpp"
#include<cuda_runtime.h>
#include<cuda.h>

using namespace std; 

// this will be used to track if some node's dist gets updated in dist array 
__device__ int d_changed = 0; 


/**
 * @brief Initilize the given array with the INT_MAX 
 * 
 * @param dist 
 * @param N 
 * @return __global__ 
 */
__global__ void dist_init( double* dist, int N ){
    int tid = blockIdx.x*blockDim.x + threadIdx.x; 
    if( tid >= N ) return ; 
    dist[tid] = INT_MAX; 
}


/**
 * @brief does the atomic mean of two values using the atmoicCAS and return the old distance
 * 
 * @param address 
 * @param val 
 * @return __device__ 
 */
__device__ double atomicMinDouble(double* address, double val)
{
    unsigned long long int* address_as_ull =
        (unsigned long long int*)address;

    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed;

    do {
        assumed = old;

        double old_val = __longlong_as_double(assumed);

        if (old_val <= val)
            break;

        old = atomicCAS(
            address_as_ull,
            assumed,
            __double_as_longlong(val)
        );

    } while (assumed != old);

    return __longlong_as_double(old);
}


/**
 * @brief 
 * 
 * @param G 
 * @param dist 
 * @param N 
 * @return __global__ 
 */
__global__ void relaxdist(CSRGraph* G, double* dist, int N)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= N || dist[tid] == INT_MAX)
        return;

    for (int i = G->row_ptr[tid];
         i < G->row_ptr[tid + 1];
         i++)
    {
        int v = G->col_ind[i];

        double newDist = dist[tid] + G->values[i];

        double oldDist = atomicMinDouble(&dist[v], newDist);

        if (newDist < oldDist)
            d_changed = 1;
    }
}



void usage(){
    cout << "Usage: ./a.out <input_file> <src> <ThreadsPerBlock" << "\n"; 
    for( int i = 0 ; i < 100;i++ ) cout << "-*-"; 
    cout << "\nThe input file's  first row should contain the number of vertices and edges\n" ; 
    cout << "\nfolloed by the three tuples of edges in the format: u v w \n"; 
}

int main( int argc, char** argv ) {
    if( argc < 2 ){
        usage(); 
        return 0;
    }
    string filename = argv[1]; 
    int src = atoi(argv[2]);
    
    CSRGraph G = readCSRGraphFromFile(filename); 

    // I am considering that -ve wights might be there so no initilization with 0 
    vector<double> h_dist(G.nodes,INT_MAX);

    // TPB and number of blocks configuratoin 
    int TPB = atoi(argv[3]); 
    int NUM_BLOCK = (G.nodes + TPB - 1 )/ TPB; 
    // devide dist initilization
    double * d_dist; 
    cudaMalloc(&d_dist,G.nodes*sizeof(double)); 
    dist_init<<<NUM_BLOCK,TPB>>>(d_dist,G.nodes); 
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK( cudaDeviceSynchronize()); 

    int h_changed = 0 ; 
    while( !h_changed ){
        // reset the d_changed to 0  
        cudaMemcpyToSymbol(d_changed,&h_changed,sizeof(int)); 

    }


    return 0 ; 
}