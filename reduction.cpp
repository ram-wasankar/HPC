// Parallel Reduction using OpenMP
#include <iostream>
#include <omp.h>          // OpenMP header
#include <climits>        // For INT_MAX, INT_MIN
using namespace std;

int main() {
  int arr[] = {3, 1, 7, 2, 9, 4, 6, 8, 5}; // Input array
  int n = 9;                               // Size of array

  int minVal = INT_MAX;                    // Start with largest possible
  int maxVal = INT_MIN;                    // Start with smallest possible
  long long sum = 0;                       // Sum starts at 0

  // OpenMP parallel reduction — splits work among threads
  #pragma omp parallel for reduction(min:minVal) \
                           reduction(max:maxVal) \
                           reduction(+:sum)
  for (int i = 0; i < n; i++) {
    if (arr[i] < minVal) minVal = arr[i]; // Each thread finds local min
    if (arr[i] > maxVal) maxVal = arr[i]; // Each thread finds local max
    sum += arr[i];                         // Each thread adds to local sum
  }
  // OpenMP merges all thread results automatically!

  double avg = (double)sum / n;           // Calculate average

  cout << "Min: " << minVal << endl;
  cout << "Max: " << maxVal << endl;
  cout << "Sum: " << sum    << endl;
  cout << "Avg: " << avg    << endl;

  return 0;
}


// Compile: g++ -fopenmp reduction.cpp -o reduction
// Run: ./reduction
// output:Min: 1 Max: 9 Sum: 45 Avg: 5