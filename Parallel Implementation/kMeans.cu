#include <bits/stdc++.h>
#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>


#define K 3
#define TPB 64
#define MAX_ITER 30

inline __device__ float distance(int x1, int y1, int x2, int y2)
{
	return sqrt((float)(x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
}

inline __global__ void kMeansClusterAssignment(int N, int* d_datapoint_x, int* d_datapoint_y, int* d_clust_assn, int* d_centroids_x, int* d_centroids_y)
{

	//get idx for this datapoint
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//bounds check
	if (idx >= N) return;

	//find the closest centroid to this datapoint
	float min_dist = INFINITY;
	int closest_centroid = 0;

	for (int c = 0; c < K; ++c)
	{
		float dist = distance(d_datapoint_x[idx], d_datapoint_y[idx], d_centroids_x[c], d_centroids_y[c]);
		// printf("%f \n", dist);
		if (dist < min_dist)
		{
			min_dist = dist;
			closest_centroid = c;
		}
	}

	//assign closest cluster id for this datapoint/thread
	d_clust_assn[idx] = closest_centroid;
}


inline __global__ void kMeansCentroidUpdate(int N, int* d_datapoints_x, int* d_datapoints_y, int* d_clust_assn, int* d_centroids_x, int* d_centroids_y, int* d_clust_sizes)
{
	//get idx of thread at grid level
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//bounds check
	if (idx >= N) return;

	//get idx of thread at the block level
	const int s_idx = threadIdx.x;

	//put the datapoints and corresponding cluster assignments in shared memory so that they can be summed by thread 0 later
	__shared__ int s_datapoints_x[TPB];
	__shared__ int s_datapoints_y[TPB];

	s_datapoints_x[s_idx] = d_datapoints_x[idx];
	s_datapoints_y[s_idx] = d_datapoints_y[idx];


	__shared__ int s_clust_assn[TPB];
	s_clust_assn[s_idx] = d_clust_assn[idx];

	__syncthreads();

	//it is the thread with idx 0 (in each block) that sums up all the values within the shared array for the block it is in
	if (s_idx == 0)
	{
		int b_clust_datapoint_sums_x[K] = { 0 };
		int b_clust_datapoint_sums_y[K] = { 0 };


		int b_clust_sizes[K] = { 0 };

		for (int j = 0; j < blockDim.x; ++j)
		{
			int clust_id = s_clust_assn[j];
			b_clust_datapoint_sums_x[clust_id] += s_datapoints_x[j];
			b_clust_datapoint_sums_y[clust_id] += s_datapoints_y[j];

			b_clust_sizes[clust_id] += 1;
		}

		//Now we add the sums to the global centroids and add the counts to the global counts.
		for (int z = 0; z < K; ++z) {

			atomicAdd(&d_centroids_x[z], b_clust_datapoint_sums_x[z]);
			atomicAdd(&d_centroids_y[z], b_clust_datapoint_sums_y[z]);

			atomicAdd(&d_clust_sizes[z], b_clust_sizes[z]);
		}
	}

	__syncthreads();

	//currently centroids are just sums, so divide by size to get actual centroids
	if (idx < K) {
		d_centroids_x[idx] = d_centroids_x[idx] / d_clust_sizes[idx];
		d_centroids_y[idx] = d_centroids_y[idx] / d_clust_sizes[idx];
	}
}


inline void find_clusters(int N, int* h_datapoints_x, int* h_datapoints_y, int* h_clust_assign)
{

	//allocate memory on the device for the data points
	int* d_datapoint_x = 0;
	int* d_datapoint_y = 0;
	//allocate memory on the device for the cluster assignments
	int* d_clust_assn = 0;
	//allocate memory on the device for the cluster centroids
	int* d_centroids_x = 0;
	int* d_centroids_y = 0;
	//allocate memory on the device for the cluster sizes
	int* d_clust_sizes = 0;


	cudaMalloc(&d_datapoint_x, N * sizeof(int));
	cudaMalloc(&d_datapoint_y, N * sizeof(int));
	cudaMalloc(&d_clust_assn, N * sizeof(int));
	cudaMalloc(&d_centroids_x, K * sizeof(int));
	cudaMalloc(&d_centroids_y, K * sizeof(int));

	cudaMalloc(&d_clust_sizes, K * sizeof(int));


	int* h_centroids_x = (int*)malloc(K * sizeof(int));
	int* h_centroids_y = (int*)malloc(K * sizeof(int));

	/*int* h_datapoints_x = (int*)malloc(N * sizeof(int));
	int* h_datapoints_y = (int*)malloc(N * sizeof(int));*/

	int* h_clust_sizes = (int*)malloc(K * sizeof(int));
	// int* h_assign = (int*)malloc(N * sizeof(int));

	//srand(time(0));

	//initialize centroids
	/*for (int d = 0; d < N; ++d) {
		h_datapoints_x[d] = rand() % 100;
		h_datapoints_y[d] = rand() % 100;
	}*/

	//Initializing centroids 
	for (int c = 0; c < K; ++c) {
		h_centroids_x[c] = h_datapoints_x[c];
		h_centroids_y[c] = h_datapoints_y[c];
	}

	/*for (int c = 0; c < K; ++c) {
		printf("%d ", h_centroids_x[c]);
	}*/

	cudaMemcpy(d_centroids_x, h_centroids_x, K * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_centroids_y, h_centroids_y, K * sizeof(int), cudaMemcpyHostToDevice);

	cudaMemcpy(d_datapoint_x, h_datapoints_y, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_datapoint_y, h_datapoints_y, N * sizeof(int), cudaMemcpyHostToDevice);

	cudaMemcpy(d_clust_sizes, h_clust_sizes, K * sizeof(int), cudaMemcpyHostToDevice);

	int cur_iter = 1;

	while (cur_iter < MAX_ITER)
	{
		//call cluster assignment kernel
		// ho gaya 
		kMeansClusterAssignment << <(N + TPB - 1) / TPB, TPB >> > (N, d_datapoint_x, d_datapoint_y, d_clust_assn, d_centroids_x, d_centroids_y);

		//copy new centroids back to host 
		cudaMemcpy(h_centroids_x, d_centroids_x, K * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_centroids_y, d_centroids_y, K * sizeof(int), cudaMemcpyDeviceToHost);

		/*for (int i = 0; i < K; ++i) {
			printf("Iteration %d: centroid %d: %d %d\n", cur_iter, i, h_centroids_x[i], h_centroids_y[i]);
		}*/

		//reset centroids and cluster sizes (will be updated in the next kernel)
		cudaMemset(d_centroids_x, 0, K * sizeof(int));
		cudaMemset(d_centroids_y, 0, K * sizeof(int));

		cudaMemset(d_clust_sizes, 0, K * sizeof(int));

		//call centroid update kernel
		kMeansCentroidUpdate << <(N + TPB - 1) / TPB, TPB >> > (N, d_datapoint_x, d_datapoint_y, d_clust_assn, d_centroids_x, d_centroids_y, d_clust_sizes);

		cur_iter += 1;
	}
	cudaMemcpy(h_clust_assign, d_clust_assn, N * sizeof(int), cudaMemcpyDeviceToHost);



	/*for (int k = 0; k < K; k++) {
		printf("\ncluster %d \n", k);
		for (int i = 0; i < N; i++) {
			if (h_clust_assign[i] == k) {
				printf("(%d,%d)-%d ", h_datapoints_x[i], h_datapoints_y[i], h_clust_assign[i]);
			}
		}
	}*/

	cudaFree(d_datapoint_x);
	cudaFree(d_datapoint_y);

	cudaFree(d_clust_assn);
	cudaFree(d_centroids_x);
	cudaFree(d_centroids_y);
	cudaFree(d_clust_sizes);

	free(h_centroids_x);
	free(h_centroids_y);

	/*free(h_datapoints_x);
	free(h_datapoints_y);*/

	free(h_clust_sizes);
	// free(h_assign);

	return;
}