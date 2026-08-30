#include<iostream> 
#include<bits/stdc++.h>
#include<cuda.h> 
#include<cuda_runtime.h>
#include "CSRGraph.hpp"
#include "../utils/utils.hpp"

using namespace std; 


/**
 * @brief Store the pointer on the device for CSR representation
 * 
 */
struct DeviceCSRGraph {
    int* row_ptr;
    int* col_ind;
    double* values;
};


/**
 * @brief Initilize the given array with the INT_MAX 
 * 
 * @param dist 
 * @param N 
 * @return __global__ 
 */
__global__ void dist_init( int* dist, int N ){
    int tid = blockIdx.x*blockDim.x + threadIdx.x; 
    if( tid >= N ) return ; 
    dist[tid] = INT_MAX; 
}


__device__ bool d_changed = false; 

__global__ void BFS( int* dist, DeviceCSRGraph g, int N ) {
	
	// push BFS 
	// get the global thread id 
	int tid = blockIdx.x* blockDim.x + threadIdx.x; 
	if( tid >= N || dist[tid] == INT_MAX) return ; 

	// find the neighbours and push the distance (minimum distance should be pushed) 
    for( int i = g.row_ptr[tid] ; i < g.row_ptr[tid+1] ; i++ ){
        
        int neighbor = g.col_ind[i];
        int new_dist = dist[tid] + 1;
        
        // atomicMin updates memory and returns the OLD value
        int old_dist = atomicMin(&dist[neighbor], new_dist); 
        
        // If the new distance is smaller than the old distance, a change was made
        if( new_dist < old_dist ){
            d_changed = true; 
        }
    }
	
	
}

void printUsage(){
    cout << "Usage: ./a.out <input_file> <src> <TPB>  "; 
    cout << "\n"; 
}


int main( int argc, char** argv){
    if( argc < 4 ){
        printUsage(); 
        return 0; 
    }
    
    string file_name = argv[1]; 
    int src = atoi(argv[2]); 
    int TPB = atoi( argv[3]); 

    // validate the src 
    CSRGraph G = readCSRGraphFromFile(file_name);
    if( src >= G.nodes){
        cout << "invalid node for given graph \n " ; 
        return 0 ; 
    }

    // get the number of block s
    int NUM_BLOCK = (G.nodes + TPB - 1 )/ TPB ; 

    // create device and host distance array and initialize them 
    vector<int> h_dist( G.nodes , 0 ); 
    int * d_dist;
    CUDA_CHECK( cudaMalloc(&d_dist, sizeof(int) * G.nodes) );
    dist_init<<<NUM_BLOCK,TPB>>>(d_dist,G.nodes); 
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK( cudaDeviceSynchronize()); 


    //create the CSR graph struct for the GPU and copy the data to the GPU 
    int *d_row_ptr, *d_col_idx;
    double *d_values; 
    CUDA_CHECK( cudaMalloc(&d_row_ptr, (G.nodes + 1)* sizeof(int))); 
    CUDA_CHECK( cudaMalloc(&d_col_idx, (G.edges)* sizeof(int))); 
    CUDA_CHECK( cudaMalloc(&d_values,  (G.edges)* sizeof(double))); 

    CUDA_CHECK( cudaMemcpy(d_row_ptr, G.row_ptr.data(), (G.nodes+1)*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK( cudaMemcpy(d_col_idx, G.col_ind.data(), G.edges*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK( cudaMemcpy(d_values, G.values.data(), G.edges*sizeof(double), cudaMemcpyHostToDevice)); 
    DeviceCSRGraph d_G{d_row_ptr,d_col_idx,d_values}; 

    int source_dist = 0;
    CUDA_CHECK(cudaMemcpy(d_dist + src, &source_dist, sizeof(int), cudaMemcpyHostToDevice));

    cout << "TPB: " << TPB << "\nNUM_BLOCKS: " << NUM_BLOCK << "\n\n"; 

    bool h_changed = true; 

    while( h_changed ) {
	h_changed = false; 
	CUDA_CHECK(cudaMemcpyToSymbol(d_changed,&h_changed, sizeof(bool))); 
	BFS<<<NUM_BLOCK, TPB>>>(d_dist,d_G,G.nodes); 
	CUDA_CHECK(cudaGetLastError()); 
	CUDA_CHECK(cudaMemcpyFromSymbol(&h_changed,d_changed,sizeof(bool))); 
	
    }

    CUDA_CHECK( cudaMemcpy(h_dist.data(), d_dist, G.nodes*sizeof(int),cudaMemcpyDeviceToHost)); 

   
    for( int i = 0 ; i < h_dist.size(); i++ ) cout << "Node : " << i << " -> " << h_dist[i] << "\n"; 

    CUDA_CHECK(cudaFree(d_values)); 
    CUDA_CHECK(cudaFree(d_col_idx)); 
    CUDA_CHECK(cudaFree(d_row_ptr)); 
    CUDA_CHECK(cudaFree(d_dist));


    return  0; 
}
