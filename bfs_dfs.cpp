// Parallel BFS and DFS using OpenMP in C++
#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>         // OpenMP header for parallelism
using namespace std;

const int MAX = 100;       // Maximum nodes in graph
vector<int> adj[MAX];     // Adjacency list for graph
bool visited[MAX];         // Track visited nodes

// ===== PARALLEL BFS =====
void bfs(int start, int n) {
  fill(visited, visited + n, false); // Reset visited array
  queue<int> q;
  visited[start] = true;             // Mark start node visited
  q.push(start);                       // Push start to queue
  cout << "BFS: ";
  while (!q.empty()) {
    int node = q.front(); q.pop();     // Get front node
    cout << node << " ";
    // Process neighbors in parallel
    #pragma omp parallel for           // OpenMP parallel loop
    for (int i = 0; i < adj[node].size(); i++) {
      int neighbor = adj[node][i];
      if (!visited[neighbor]) {
        #pragma omp critical             // Only one thread at a time here
        {
          if (!visited[neighbor]) {
            visited[neighbor] = true;
            q.push(neighbor);
          }
        }
      }
    }
  }
  cout << endl;
}

// ===== PARALLEL DFS =====
void dfs(int node) {
  visited[node] = true;               // Mark current node visited
  cout << node << " ";
  // Visit all unvisited neighbors in parallel
  #pragma omp parallel for
  for (int i = 0; i < adj[node].size(); i++) {
    if (!visited[adj[node][i]]) {
      #pragma omp critical               // Prevent data race
      dfs(adj[node][i]);               // Recursively visit neighbor
    }
  }
}

int main() {
  int n = 6;                           // 6 nodes (0 to 5)
  // Add edges (undirected graph)
  adj[0].push_back(1); adj[1].push_back(0);
  adj[0].push_back(2); adj[2].push_back(0);
  adj[1].push_back(3); adj[3].push_back(1);
  adj[1].push_back(4); adj[4].push_back(1);
  adj[2].push_back(5); adj[5].push_back(2);

  bfs(0, n);                           // BFS from node 0
  fill(visited, visited + n, false); // Reset for DFS
  cout << "DFS: ";
  dfs(0);                              // DFS from node 0
  cout << endl;
  return 0;
}


// Compile: g++ -fopenmp bfs_dfs.cpp -o bfs_dfs
// Run: ./bfs_dfs
// output: BFS: 0 1 2 3 4 5 DFS: 0 1 3 4 2 5