#pragma once 
#include<bits/stdc++.h>

using namespace std;

/**
 * @brief The struct to store the graphs in the CSR format 
 * 
 */
struct CSRGraph{
    int nodes; 
    int edges; 
    vector<int> row_ptr; 
    vector<int> col_ind; 
    vector<double> values; 
}; 


CSRGraph createCSRGraph( int V, vector<vector<pair<int,double>>>& adjList){
    CSRGraph graph; 
    graph.nodes = V; 
    graph.edges = 0;
    graph.row_ptr.resize(V+1,0);

    // the row_ptr array is nothing but the prefix sum of number of edges in each row. 
    for( int i = 0 ; i < V; i++ ){
        graph.row_ptr[i+1] = graph.row_ptr[i] + adjList[i].size();
    }
    graph.edges = graph.row_ptr[V];

    graph.col_ind.resize(graph.edges);
    graph.values.resize(graph.edges); 

    // copy edges in to the values and col_ind arrays;
    for( int i = 0 ; i < V; i++ ){
        for( int j = graph.row_ptr[i] ; j < graph.row_ptr[i+1]; j++ ){
            graph.col_ind[j] = adjList[i][j-graph.row_ptr[i]].first; 
            graph.values[j] = adjList[i][j-graph.row_ptr[i]].second;
        }
    }

    return graph ; 
}


/**
 * @brief read the graph from a file and return the CSRGraph representation of it. 
 *        the first line in the file should contain the number of vertices and edges. 
 * @param filename 
 * @return CSRGraph 
 */
CSRGraph readCSRGraphFromFile( string filename){
    ifstream file(filename); 
    int V,E; 
    file >> V >> E; 

    vector<vector<pair<int,double>>> adjList(V); 

    for( int i = 0 ; i < E; i++ ){
        int u,v; 
        double w; 
        file >> u >> v >> w; 
        adjList[u].push_back({v,w}); 
    }

    return createCSRGraph(V,adjList);   
}
