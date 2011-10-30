/**
 * @File clumper_kernels.cu GPGPU K-Medoid Clustering Implementation.
 * @author Rodrigo González del Cueto, Copyright (C) 2011.
 * @see The GNU GENERAL PUBLIC LICENSE Version 2 (GPL)
 *
 * Clumper is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * Clumper is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Clumper. If not, see <http://www.gnu.org/licenses/>.
 */

#include "clumper.h"

// GPU Kernels Shared Memory Pointer.
extern __shared__ float shared_memory[];

__global__ void associate_closest_medoid ( const unsigned int clusters,
					   const unsigned int elements,
					   const unsigned int medoid_indexes[],
					   unsigned int medoid_assoc[],
					   const float distance_matrix[] )
{
  unsigned int id = threadIdx.x;
  while ( id < elements )
    {
      unsigned int best_medoid_index = medoid_indexes [ 0 ];
      float best_medoid_distance =
      	distance_matrix [ elements * medoid_indexes [ 0 ] + id ];

      unsigned int i = 1;
      while ( i < clusters )
	{
	  unsigned int current_index = medoid_indexes [ i ];
	  float new_candidate_cost =
	    distance_matrix [ elements * current_index + id ];

	  if ( new_candidate_cost < best_medoid_distance )
	    {
	      best_medoid_distance = new_candidate_cost;
	      best_medoid_index = current_index;
	    }
	  i++;
	}
      medoid_assoc [ id ] = best_medoid_index;
      id += blockDim.x;
    }
}

__global__ void reduce_medoid_candidates ( const unsigned int clusters,
					   const unsigned int elements,
					   const unsigned int medoid_assoc[],
					   const float distance_matrix[],
					   float medoid_candidates_cost[],
					   unsigned int medoid_indexes[],
					   float medoid_costs[],
					   int *diff_flag )
{
  float * medoid_cost_local_reduction = ( float * ) shared_memory;

  unsigned int * medoid_index_local_reduction =
    ( unsigned int * ) &medoid_cost_local_reduction [ elements ];

  unsigned int id = threadIdx.x;
  while ( id < elements )
    {
      medoid_candidates_cost [ id ] = 0;
      id += gridDim.x;
    }
  __syncthreads();

  // Get each candidate cost per cluster
  for ( unsigned int curr_cluster = 0; curr_cluster < clusters; curr_cluster++ )
    {
      unsigned int curr_cluster_index = medoid_indexes [ curr_cluster ];

      id = threadIdx.x;
      while ( id < elements )
	{
	  if ( medoid_candidates_cost [ id ] != CUDART_INF_F )
	    {
	      float candidate_cost = 0;
	      for ( unsigned int i = 0; i < elements; i++ )
		{
		  if ( medoid_assoc [ i ] == curr_cluster_index )
		    {
		      // Version A
		      candidate_cost += distance_matrix [ elements * i + id ];
		      // Version B
		      //float d = distance_matrix [ elements * i + id ];
		      //candidate_medoid_cost += d * d;
		    }
		}
	      medoid_candidates_cost [ id ] = candidate_cost;
	    }
	  id += blockDim.x;
	}

      // Over-block Reduction ( Device Max-threads < elements )
      float a, b;
      unsigned int best_idx;

      a = medoid_candidates_cost [ threadIdx.x ];
      best_idx = threadIdx.x;

      unsigned int id = threadIdx.x + gridDim.x;
      while ( id < elements )
	{
	  b = medoid_candidates_cost [ id ];

	  if ( b < a )
	    {
	      a = b;
	      best_idx = id;
	    }
	  id += gridDim.x;
	}

      medoid_cost_local_reduction [ threadIdx.x ] = a;
      medoid_index_local_reduction [ threadIdx.x ] = best_idx;

      __syncthreads();
      
      // Block Reduction
      {
	int reduce_idx = 1;
	const int reduce_limit = blockDim.x;

	while ( reduce_idx < reduce_limit ) reduce_idx <<= 1;
	float a, b;
	while ( reduce_idx != 0 )
	  {
	    if ( threadIdx.x < reduce_idx &&
		 threadIdx.x + reduce_idx < reduce_limit )
	      {
		a = medoid_cost_local_reduction [ threadIdx.x ];
		b = medoid_cost_local_reduction [ threadIdx.x + reduce_idx ];

		if ( b < a )
		  {
		    medoid_cost_local_reduction [ threadIdx.x ] = b;
		    medoid_index_local_reduction [ threadIdx.x ] =
		      medoid_index_local_reduction [ threadIdx.x + reduce_idx ];
		  }
	      }
	    __syncthreads ();
	    reduce_idx >>= 1;
	  }

	if ( threadIdx.x == 0 )
	  {
	    unsigned int old_medoid_idx = medoid_indexes [ curr_cluster ];
	    unsigned int best_candidate_idx = medoid_index_local_reduction [ 0 ];

	    if ( old_medoid_idx != best_candidate_idx )
	      *diff_flag |= 1;

	    medoid_indexes [ curr_cluster ] = best_candidate_idx;
	    medoid_candidates_cost [ best_candidate_idx ] = CUDART_INF_F;
	    medoid_costs [ curr_cluster ] = medoid_cost_local_reduction [ 0 ];
	  }
      }      
    }
  return;
}

void k_medoid_clustering ( const unsigned int clusters,
			   const float distance_matrix[],
			   const unsigned int elements,
			   const unsigned int max_iter )
{

  unsigned int iterations_limit;

  if ( max_iter == 0 )
    iterations_limit = std::numeric_limits<unsigned int>::infinity();
  else
    iterations_limit = max_iter;

  // Random Seed Initialization
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  //clock_gettime(CLOCK_REALTIME, &ts);
  srand48( ts.tv_nsec );

  // Get Max Threads per Block on Default Device
  cudaDeviceProp prop;
  CUDA_SAFE_CALL ( cudaGetDeviceProperties( &prop, 0 ) );
  const unsigned int threads = elements <= prop.maxThreadsPerBlock ?
    elements : prop.maxThreadsPerBlock;

  std::cout << "Using " << threads << " threads.\n";

  // Device Memory Pointers
  float *dev_distance_matrix_ptr;          // Distance Matrix

  float *dev_medoids_costs_ptr;             // Selected Medoids' Cost
  unsigned int *dev_medoids_indexes_ptr;    // Selected Medoids' Indexes

  float *dev_medoid_candidates_cost_ptr;   // Every Possible Medoid Cost

  unsigned int *dev_medoid_assoc_ptr;      // New Medoid Association Table

  int *dev_diff_flag_ptr;                  // Difference Flag

  unsigned int *host_best_cluster_medoid_assoc_ptr;
  float host_best_cluster_cost = std::numeric_limits<float>::infinity();

  // Device Memory Allocation
  CUDA_SAFE_CALL( cudaMalloc( ( void ** ) &dev_distance_matrix_ptr,
			      sizeof ( float ) * elements * elements ) );

  CUDA_SAFE_CALL( cudaMalloc( ( void ** ) &dev_medoids_costs_ptr,
			      sizeof ( float ) * clusters ) );

  CUDA_SAFE_CALL( cudaMalloc( ( void ** ) &dev_medoids_indexes_ptr,
			      sizeof ( unsigned int ) * clusters ) );

  CUDA_SAFE_CALL( cudaMalloc( ( void ** ) &dev_medoid_candidates_cost_ptr,
			      sizeof ( float ) * elements ) );
  
  CUDA_SAFE_CALL( cudaMalloc( ( void ** ) &dev_medoid_assoc_ptr,
			      sizeof ( unsigned int ) * elements ) );

  CUDA_SAFE_CALL( cudaMalloc( ( void ** ) &dev_diff_flag_ptr,
			      sizeof ( int ) ) );
  
  // Host to Device Data Initialization
  CUDA_SAFE_CALL( cudaMemcpy ( dev_distance_matrix_ptr,
			       distance_matrix,
			       sizeof ( float ) * elements * elements,
			       cudaMemcpyHostToDevice ) );
  
  // Medoid Set Copy to Device
  {
    // Initialize with Random Medoids
    std::set<unsigned int> init_medoid_indexes;
    while ( init_medoid_indexes.size() < clusters )
      {
	unsigned int rand_medoid =
	  (unsigned int) ( ceil( drand48() * elements ) - 1 );
	
	init_medoid_indexes.insert( rand_medoid );
      }

    unsigned int host_medoid_indexes [ clusters ];
    unsigned int i = 0;
    for ( std::set<unsigned int>::iterator it = init_medoid_indexes.begin();
	  it != init_medoid_indexes.end();
	  ++it )
      {
	host_medoid_indexes [ i++ ] = *it;
      }

    // Delete Initialization Medoid Set
    init_medoid_indexes.clear();

    // Copy Medoid Indexes into Device
    CUDA_SAFE_CALL( cudaMemcpy ( dev_medoids_indexes_ptr,
				 &host_medoid_indexes,
				 sizeof ( unsigned int ) * clusters,
				 cudaMemcpyHostToDevice ) );
  }

  // Check Medoid Set Copy
#ifdef _DEBUG
  {
    unsigned int host_medoid_indexes [ clusters ];
    CUDA_SAFE_CALL( cudaMemcpy( &host_medoid_indexes,
				dev_medoids_indexes_ptr,
				sizeof ( unsigned int ) * clusters,
				cudaMemcpyDeviceToHost ) );

    std::cout << "Initial Medoid Indexes\n";
    for ( unsigned int i = 0; i < clusters; i++ )
      {
    	// Output Device Medoid Indexes.
    	std::cout << i << ": " << host_medoid_indexes [ i ] << "\n";
      }
    std::cout << std::endl;
  }
#endif

  // K Medoid Clustering Loop
  unsigned int iterations = 0;
  int host_diff_flag;

  do
    {
      host_diff_flag = 0;
      CUDA_SAFE_CALL( cudaMemcpy( dev_diff_flag_ptr,
				  &host_diff_flag,
 				  sizeof ( int ),
				  cudaMemcpyHostToDevice ) );

      if ( max_iter == 0 )
	std::cout << "Iteration: " << ( iterations + 1 ) << std::endl;
      else
	std::cout << "Iteration: " << ( iterations + 1 ) << " / " << max_iter << std::endl;

      // Associate each datapoint with closest medoid
      associate_closest_medoid
	<<< 1, threads >>> ( clusters,
			     elements,
			     dev_medoids_indexes_ptr,
			     dev_medoid_assoc_ptr,
			     dev_distance_matrix_ptr );

      // Check Device Current Medoid Association Table. DEBUG
#ifdef _DEBUG
      {
	// Copy Medoid Set into Host
	unsigned int host_medoid_indexes [ clusters ];
	CUDA_SAFE_CALL( cudaMemcpy( &host_medoid_indexes,
				    dev_medoids_indexes_ptr,
				    sizeof ( unsigned int ) * clusters,
				    cudaMemcpyDeviceToHost ) );

	// Copy Medoid Assoc Table
	unsigned int host_medoid_assoc [ elements ];
	CUDA_SAFE_CALL( cudaMemcpy( &host_medoid_assoc,
				    dev_medoid_assoc_ptr,
				    sizeof ( unsigned int ) * elements,
				    cudaMemcpyDeviceToHost ) );

	std::cout << "Assoc Medoid | Element Idx\n";
	for ( unsigned int i = 0; i < clusters; i++ )
	  {
	    unsigned int assoc_it = host_medoid_indexes [ i ];
	    for ( unsigned int j = 0; j < elements; j++ )
	      {
		if ( host_medoid_assoc [ j ] == assoc_it )
		  {
		    std::cout <<
		      assoc_it << " | " << j << std::endl;
		  }
	      }
	  }
	std::cout << "------------------------\n";
      }
#endif

      // Update Medoids
      reduce_medoid_candidates
	<<< 1, threads,	( sizeof ( float ) + sizeof ( unsigned int ) ) * elements >>>
	( clusters,
	  elements,
	  dev_medoid_assoc_ptr,
	  dev_distance_matrix_ptr,
	  dev_medoid_candidates_cost_ptr,
	  dev_medoids_indexes_ptr,
	  dev_medoids_costs_ptr,
	  dev_diff_flag_ptr );

      // Check Device Medoid Partial cluster Cost. DEBUG
#ifdef _DEBUG
      {
	// Copy Medoid Set into Host
	unsigned int host_medoid_indexes [ clusters ];
	CUDA_SAFE_CALL( cudaMemcpy( &host_medoid_indexes,
				    dev_medoids_indexes_ptr,
				    sizeof ( unsigned int ) * clusters,
				    cudaMemcpyDeviceToHost ) );

	float host_medoids_costs [ clusters ];
	CUDA_SAFE_CALL( cudaMemcpy( &host_medoids_costs,
				    dev_medoids_costs_ptr,
				    sizeof ( float ) * clusters,
				    cudaMemcpyDeviceToHost ) );

	std::cout << "\nUpdated Medoid: Element Idx | Medoid Cost\n";
	for ( unsigned int i = 0; i < clusters; i++ )
	  {
	    std::cout <<
	      i << " | " <<
	      host_medoid_indexes [ i ] << " | " <<
	      host_medoids_costs [ i ] << std::endl;
	  }
	std::cout << std::endl;
      }
#endif

      // Check Current Clustering Cost.
#ifdef _DEBUG
      {
	// Copy New Medoid Set into Host
	unsigned int host_medoid_indexes [ clusters ];
	CUDA_SAFE_CALL( cudaMemcpy( &host_medoid_indexes,
				    dev_medoids_indexes_ptr,
				    sizeof ( unsigned int ) * clusters,
				    cudaMemcpyDeviceToHost ) );

	float host_medoids_costs [ elements ];
	CUDA_SAFE_CALL( cudaMemcpy( &host_medoids_costs,
				    dev_medoids_costs_ptr,
				    sizeof ( float ) * clusters,
				    cudaMemcpyDeviceToHost ) );
	
	float cluster_score = 0.0f;

	for ( unsigned int i = 0; i < clusters; i++ )
	  {
	    cluster_score += host_medoids_costs [ i ];
	  }

	std::cout << "Clustering Score\n";
	std::cout << cluster_score << "\n";
	std::cout << std::endl;
      }
#endif

#ifdef _DEBUG
	{
	// Copy Medoid Set into Host
	unsigned int host_medoid_indexes [ clusters ];
	CUDA_SAFE_CALL( cudaMemcpy( &host_medoid_indexes,
				    dev_medoids_indexes_ptr,
				    sizeof ( unsigned int ) * clusters,
				    cudaMemcpyDeviceToHost ) );

	// Copy Medoid Assoc Table
	unsigned int host_medoid_assoc [ elements ];
	CUDA_SAFE_CALL( cudaMemcpy( &host_medoid_assoc,
				    dev_medoid_assoc_ptr,
				    sizeof ( unsigned int ) * elements,
				    cudaMemcpyDeviceToHost ) );

	//Check Medoid Sets. DEBUG 
	std::cout << "Index | New Assoc Tables\n";
	for ( unsigned int i = 0; i < clusters; i++ )
	  {
	    unsigned int assoc_it = host_medoid_indexes [ i ];
	    for ( unsigned int j = 0; j < elements; j++ )
	      {
		if ( host_medoid_assoc [ j ] == assoc_it )
		  {
		    std::cout <<
		      assoc_it << " | " << j << std::endl;
		  }
	      }
	  }
	  std::cout << std::endl;
	}
#endif

	CUDA_SAFE_CALL( cudaMemcpy( &host_diff_flag,
				    dev_diff_flag_ptr,
				    sizeof ( int ),
				    cudaMemcpyDeviceToHost ) );

#ifdef _DEBUG
	{
          std::cout << "Diff Flag = " << host_diff_flag << std::endl;
	  if ( host_diff_flag )
	    std::cout << "Medoids changed!\n";
	  else
	    std::cout << "No change in medoids, terminating...\n";
	}
#endif


    } while ( host_diff_flag && ( ++iterations < iterations_limit ) );
  // End Clustering Loop

  // Print Results
  // TODO: Standarize Output format
  if ( iterations < iterations_limit )
    std::cout << "Converged after " << ( iterations + 1 ) << " iterations." << std::endl;
  else
    std::cout << "Reached iteration limit @ " << ( iterations ) << " / " << iterations_limit << "." << std::endl;

  // Copy Results from Device
  unsigned int host_medoid_indexes [ clusters ];
  CUDA_SAFE_CALL( cudaMemcpy( &host_medoid_indexes,
	dev_medoids_indexes_ptr,
	sizeof ( unsigned int ) * clusters,
	cudaMemcpyDeviceToHost ) );

  float host_medoids_costs [ elements ];
  CUDA_SAFE_CALL( cudaMemcpy( &host_medoids_costs,
	dev_medoids_costs_ptr,
	sizeof ( float ) * clusters,
	cudaMemcpyDeviceToHost ) );

  unsigned int host_medoid_assoc [ elements ];
  CUDA_SAFE_CALL( cudaMemcpy( &host_medoid_assoc,
			      dev_medoid_assoc_ptr,
			      sizeof ( unsigned int ) * elements,
			      cudaMemcpyDeviceToHost ) );
	
  float cluster_score = 0.0f;

  for ( unsigned int i = 0; i < clusters; i++ )
    {
      cluster_score += host_medoids_costs [ i ];
    }

  std::cout << "Clustering Score\n";
  std::cout << cluster_score << "\n";
  std::cout << std::endl;

  std::cout << "Resulting Medoid Set\n";

  unsigned int n = 0;
  for ( unsigned int i = 0; i < clusters; i++ )
    {
      std::cout << "#Cluster\t" << ++n << std::endl;
      unsigned int medoid_index = host_medoid_indexes[ i ];
      for ( unsigned int j = 0; j < elements; j++ )
  	{
  	  if ( medoid_index == host_medoid_assoc [ j ] )
  	    std::cout << ( j + 1 ) << "\n";
  	}
    }

  // Device Memory Deallocation
  CUDA_SAFE_CALL( cudaFree ( dev_distance_matrix_ptr ) );
  CUDA_SAFE_CALL( cudaFree ( dev_medoid_candidates_cost_ptr ) );
  CUDA_SAFE_CALL( cudaFree ( dev_medoids_costs_ptr ) );
  CUDA_SAFE_CALL( cudaFree ( dev_medoids_indexes_ptr ) );
  CUDA_SAFE_CALL( cudaFree ( dev_medoid_assoc_ptr ) );
  CUDA_SAFE_CALL( cudaFree ( dev_diff_flag_ptr ) );

  return;
}
