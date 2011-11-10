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

__global__ void get_current_medoids_cost ( const unsigned int clusters,
					   const unsigned int elements,
					   const unsigned int medoid_indexes[],
					   const unsigned int medoid_assoc[],
					   const float distance_matrix[],
					   float medoids_costs[] )
{
  float * medoid_cost_local_reduction = ( float * ) shared_memory;

  for ( unsigned int i = 0; i < clusters; i++ )
    {
      unsigned int medoid_idx = medoid_indexes [ i ];
      float partial_cost = 0.0f;

      // Over-block Reduction ( Device Max-threads < elements )
      unsigned int id = threadIdx.x;
      while ( id < elements )
	{
	  if ( medoid_assoc [ id ] == medoid_idx )
	    {
	      partial_cost += distance_matrix [ medoid_idx * elements + id ];
	    }
	  id += blockDim.x;
	}

      medoid_cost_local_reduction [ threadIdx.x ] = partial_cost;
      __syncthreads();
      
      // Block Reduction
      int reduce_idx = 1;
      const int reduce_limit = blockDim.x;

      while ( reduce_idx < reduce_limit ) reduce_idx <<= 1;
      while ( reduce_idx != 0 )
	{
	  if ( threadIdx.x < reduce_idx &&
	       threadIdx.x + reduce_idx < reduce_limit )
	    {
	      medoid_cost_local_reduction [ threadIdx.x ] +=
		medoid_cost_local_reduction [ threadIdx.x + reduce_idx ];
	    }
	  __syncthreads ();
	  reduce_idx >>= 1;
	}

      if ( threadIdx.x == 0 )
	{
	  medoids_costs [ i ] = medoid_cost_local_reduction [ 0 ];
	} 
    }
  return;
}

__global__ void associate_closest_medoid ( const unsigned int clusters,
					   const unsigned int elements,
					   const unsigned int medoid_indexes[],
					   unsigned int medoid_assoc[],
					   const float distance_matrix[] )
{
  // Over-block Reduction ( Device Max-threads < elements )
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

  return;
}

__global__ void clear_medoid_candidates ( const unsigned int elements,
					  float medoid_candidates_cost[] )
{
  // Over-block Reduction ( Device Max-threads < elements )
  unsigned int id = threadIdx.x;
  while ( id < elements )
    {
      medoid_candidates_cost [ id ] = 0.0f;
      id += blockDim.x;
    }
  return;
}

__global__ void compute_medoid_candidates ( const unsigned int cluster_id,
					    const unsigned int elements,
					    const unsigned int medoid_assoc[],
					    const unsigned int medoid_indexes[],
					    const float distance_matrix[],
					    float medoid_candidates_cost[] )
{
  // Over-block Reduction ( Device Max-threads < elements )
  unsigned int curr_medoid_idx = medoid_indexes [ cluster_id ];
  unsigned int id = threadIdx.x;
  while ( id < elements )
    {
      if ( medoid_candidates_cost [ id ] != CUDART_INF_F )
	{
	  float cost = 0.0f;
	  for ( unsigned int i = 0; i < elements; i++ )
	    {
	      if ( medoid_assoc [ i ] == curr_medoid_idx )
		cost += distance_matrix [ i * elements + id ];
	    }
	  medoid_candidates_cost [ id ] = cost;
	}
      id += blockDim.x;
    }
  return;
}

__global__ void reduce_medoid_candidates ( const unsigned int cluster_id,
					   const unsigned int elements,
					   unsigned int medoid_assoc[],
					   float medoid_candidates_cost[],
					   unsigned int medoid_indexes[],
					   float medoid_costs[],
					   int *diff_flag )
{
  float * medoid_cost_local_reduction = ( float * ) shared_memory;

  unsigned int * medoid_index_local_reduction =
    ( unsigned int * ) &medoid_cost_local_reduction [ blockDim.x ];

  // Over-block Reduction ( Device Max-threads < elements )
  float a, b;
  unsigned int best_idx;

  a = medoid_candidates_cost [ threadIdx.x ];
  best_idx = threadIdx.x;

  unsigned int id = threadIdx.x + blockDim.x;
  while ( id < elements )
    {
      b = medoid_candidates_cost [ id ];

      if ( b < a )
	{
	  a = b;
	  best_idx = id;
	}
      id += blockDim.x;
    }

  medoid_cost_local_reduction [ threadIdx.x ] = a;
  medoid_index_local_reduction [ threadIdx.x ] = best_idx;

  __syncthreads();
      
  // Block Reduction
  int reduce_idx = 1;
  const int reduce_limit = blockDim.x;

  while ( reduce_idx < reduce_limit ) reduce_idx <<= 1;
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
      unsigned int old_medoid_idx = medoid_indexes [ cluster_id ];
      unsigned int best_candidate_idx = medoid_index_local_reduction [ 0 ];

      if ( old_medoid_idx != best_candidate_idx )
	{
	  medoid_assoc [ best_candidate_idx ] = medoid_indexes [ cluster_id ];
	  medoid_indexes [ cluster_id ] = best_candidate_idx;

	  // Blacklist element in following clusters.
	  medoid_candidates_cost [ best_candidate_idx ] = CUDART_INF_F;

	  medoid_costs [ cluster_id ] = medoid_cost_local_reduction [ 0 ];
	  
	  *diff_flag |= 1;
	}
    }
  return;
}

void k_medoid_clustering ( const unsigned int clusters,
			   const float distance_matrix[],
			   const unsigned int elements,
			   const unsigned int max_iterations,
			   const unsigned int repetitions,
			   const unsigned long seed,
			   std::stringstream &output )
{

  unsigned int iterations_limit;

  if ( max_iterations == 0 )
    {
      iterations_limit = std::numeric_limits<unsigned int>::infinity();
    }
  else
    {
      iterations_limit = max_iterations;
    }

  // Random Seed Initialization
  if ( seed )
    {
      srand48( seed );
    }
  else
    {
      timespec ts;
      clock_gettime(CLOCK_MONOTONIC, &ts);
      //clock_gettime(CLOCK_REALTIME, &ts);
      srand48( ts.tv_nsec );
    }

  // Get Max Threads per Block on Default Device
  cudaDeviceProp prop;
  CUDA_SAFE_CALL ( cudaGetDeviceProperties( &prop, 0 ) );
  const unsigned int threads = elements <= prop.maxThreadsPerBlock ?
    elements : prop.maxThreadsPerBlock;

  std::cout << "Using " << threads << " threads.\n";

  // Host Variables
  std::set<unsigned int> init_medoid_indexes;

  float host_best_cluster_cost =
    std::numeric_limits<float>::infinity();    // Best Solution Found

  float *host_medoids_costs_ptr = ( float* )
    calloc ( clusters, sizeof ( float ) );     // Host Medoids' Costs

  float *host_candidates_costs_ptr = ( float* )
    calloc ( elements, sizeof ( float ) );     // Host Candidates' Costs

  unsigned int *host_medoid_assoc_ptr = ( unsigned int* )
    calloc ( elements, sizeof ( unsigned int ) ); // Host Medoid Association Table

  unsigned int *host_medoid_indexes_ptr = ( unsigned int* )
    calloc ( clusters, sizeof ( unsigned int ) ); // Host Medoid Indexes

  // Device Memory Pointers
  float *dev_distance_matrix_ptr;              // Distance Matrix

  float *dev_medoids_costs_ptr;                // Selected Medoids' Costs
  unsigned int *dev_new_medoids_indexes_ptr;   // Selected Medoids' Indexes
  unsigned int *dev_best_medoids_indexes_ptr;  // Best Medoids' Indexes

  float *dev_medoid_candidates_cost_ptr;       // Every Possible Medoid Cost

  unsigned int *dev_medoid_assoc_ptr;          // New Medoid Association Table

  int *dev_diff_flag_ptr;                      // Difference Flag

  // Device Memory Allocation
  CUDA_SAFE_CALL( cudaMalloc( ( void ** ) &dev_distance_matrix_ptr,
			      sizeof ( float ) * elements * elements ) );

  CUDA_SAFE_CALL( cudaMalloc( ( void ** ) &dev_medoids_costs_ptr,
			      sizeof ( float ) * clusters ) );

  CUDA_SAFE_CALL( cudaMalloc( ( void ** ) &dev_new_medoids_indexes_ptr,
			      sizeof ( unsigned int ) * clusters ) );

  CUDA_SAFE_CALL( cudaMalloc( ( void ** ) &dev_best_medoids_indexes_ptr,
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

  unsigned int runs = 0;
  do
    {  
      // Medoid Set Copy to Device
      // Initialize with Random Medoids
      while ( init_medoid_indexes.size() < clusters )
	{
	  unsigned int rand_medoid =
	    (unsigned int) ( ceil( drand48() * elements ) - 1 );
	
	  init_medoid_indexes.insert( rand_medoid );
	}

      unsigned int i = 0;
      for ( std::set<unsigned int>::iterator it = init_medoid_indexes.begin();
	    it != init_medoid_indexes.end();
	    ++it )
	{
	  host_medoid_indexes_ptr [ i++ ] = *it;
	}

      // Delete Initialization Medoid Set
      init_medoid_indexes.clear();

      // Copy Medoid Indexes into Device
      CUDA_SAFE_CALL( cudaMemcpy ( dev_new_medoids_indexes_ptr,
				   host_medoid_indexes_ptr,
				   sizeof ( unsigned int ) * clusters,
				   cudaMemcpyHostToDevice ) );

      // Check Medoid Set Copy
#ifdef _DEBUG
      {
	CUDA_SAFE_CALL( cudaMemcpy( host_medoid_indexes_ptr,
				    dev_new_medoids_indexes_ptr,
				    sizeof ( unsigned int ) * clusters,
				    cudaMemcpyDeviceToHost ) );

	std::cout << "Initial Medoid Indexes\n";
	for ( unsigned int i = 0; i < clusters; i++ )
	  {
	    // Output Device Medoid Indexes.
	    std::cout << i << ": " << host_medoid_indexes_ptr [ i ] << "\n";
	  }
	std::cout << std::endl;
      }
#endif

      // K Medoid Clustering Loop
      unsigned int iterations = 0;
      int host_diff_flag;
      float cluster_score;

      do
	{
	  // Reset Diff Flag
	  CUDA_SAFE_CALL( cudaMemset( dev_diff_flag_ptr, 0, sizeof ( int ) ) );

#ifdef _DEBUG
	  {
	    if ( max_iterations == 0 )
	      std::cout << "Iteration: " << ( iterations + 1 ) << std::endl;
	    else
	      std::cout << "Iteration: " << ( iterations + 1 ) << " / " <<
		max_iterations << std::endl;
	  }
#endif

	  // Associate each datapoint with closest medoid
	  associate_closest_medoid
	    <<< 1, threads >>> ( clusters,
				 elements,
				 dev_new_medoids_indexes_ptr,
				 dev_medoid_assoc_ptr,
				 dev_distance_matrix_ptr );

	  if ( iterations == 0 )
	    {
	      get_current_medoids_cost
		<<< 1, threads, sizeof ( float ) * threads >>>
		( clusters,
		  elements,
		  dev_new_medoids_indexes_ptr,
		  dev_medoid_assoc_ptr,
		  dev_distance_matrix_ptr,
		  dev_medoids_costs_ptr );

	      CUDA_SAFE_CALL( cudaMemcpy( host_medoids_costs_ptr,
					  dev_medoids_costs_ptr,
					  sizeof ( float ) * clusters,
					  cudaMemcpyDeviceToHost ) );

	      cluster_score = 0.0f;
	      for ( unsigned int i = 0; i < clusters; i++ )
		{
		  cluster_score += host_medoids_costs_ptr [ i ];
		}

	      if ( cluster_score < host_best_cluster_cost )
		{
		  host_best_cluster_cost = cluster_score;
		  CUDA_SAFE_CALL( cudaMemcpy( dev_best_medoids_indexes_ptr,
					      dev_new_medoids_indexes_ptr,
					      sizeof ( unsigned int ) * clusters,
					      cudaMemcpyDeviceToDevice ) );
		}
	    }

	  // Check Device Current Medoid Association Table. DEBUG
#ifdef _DEBUG
	  {
	    // Copy Medoid Set into Host
	    CUDA_SAFE_CALL( cudaMemcpy( host_medoid_indexes_ptr,
					dev_new_medoids_indexes_ptr,
					sizeof ( unsigned int ) * clusters,
					cudaMemcpyDeviceToHost ) );

	    // Copy Medoid Assoc Table
	    CUDA_SAFE_CALL( cudaMemcpy( host_medoid_assoc_ptr,
					dev_medoid_assoc_ptr,
					sizeof ( unsigned int ) * elements,
					cudaMemcpyDeviceToHost ) );

	    std::cout << "Assoc Medoid | Element Idx\n";
	    for ( unsigned int i = 0; i < clusters; i++ )
	      {
		unsigned int assoc_it = host_medoid_indexes_ptr [ i ];
		for ( unsigned int j = 0; j < elements; j++ )
		  {
		    if ( host_medoid_assoc_ptr [ j ] == assoc_it )
		      {
			std::cout <<
			  assoc_it << " | " << j << std::endl;
		      }
		  }
	      }
	    std::cout << "------------------------\n";
	  }
#endif

	  // Clear New Iteration Memory
	  clear_medoid_candidates
	    <<< 1, threads >>>
	    ( elements,
	      dev_medoid_candidates_cost_ptr );

	  // Compute new candidates
	  for ( unsigned int cluster_it = 0; cluster_it < clusters; cluster_it++ )
	    {
	      compute_medoid_candidates
		<<< 1, threads >>>
		( cluster_it,
		  elements,
		  dev_medoid_assoc_ptr,
		  dev_new_medoids_indexes_ptr,
		  dev_distance_matrix_ptr,
		  dev_medoid_candidates_cost_ptr );

#ifdef _DEBUG
	      {
		CUDA_SAFE_CALL( cudaMemcpy( host_candidates_costs_ptr,
					    dev_medoid_candidates_cost_ptr,
					    sizeof ( float ) * elements,
					    cudaMemcpyDeviceToHost ) );

		std::cout << "Cluster " << cluster_it << " candidates' costs:\n";
		for ( unsigned int i = 0; i < elements; i++ )
		  {
		    std::cout << i << ": " << host_candidates_costs_ptr [ i ] << std::endl;
		  }
		std::cout << std::endl;
	      }
#endif
	      reduce_medoid_candidates
		<<< 1, threads, ( sizeof ( float ) + sizeof ( unsigned int ) ) * threads >>>
		( cluster_it,
		  elements,
		  dev_medoid_assoc_ptr,
		  dev_medoid_candidates_cost_ptr,
		  dev_new_medoids_indexes_ptr,
		  dev_medoids_costs_ptr,
		  dev_diff_flag_ptr );
	    }

	  // Check Device Medoid Partial cluster Cost. DEBUG
#ifdef _DEBUG
	  {
	    // Copy Medoid Set into Host
	    CUDA_SAFE_CALL( cudaMemcpy( host_medoid_indexes_ptr,
					dev_new_medoids_indexes_ptr,
					sizeof ( unsigned int ) * clusters,
					cudaMemcpyDeviceToHost ) );

	    CUDA_SAFE_CALL( cudaMemcpy( host_medoids_costs_ptr,
					dev_medoids_costs_ptr,
					sizeof ( float ) * clusters,
					cudaMemcpyDeviceToHost ) );

	    std::cout << "\nUpdated Medoids\n";
	    std::cout << "Medoid | Idx | Cost\n";
	    for ( unsigned int i = 0; i < clusters; i++ )
	      {
		std::cout <<
		  i << ": " <<
		  host_medoid_indexes_ptr [ i ] << " | " <<
		  host_medoids_costs_ptr [ i ] << std::endl;
	      }
	    std::cout << std::endl;
	  }
#endif

	  // Compute Current Iteration Cluster Score
	  CUDA_SAFE_CALL( cudaMemcpy( host_medoids_costs_ptr,
				      dev_medoids_costs_ptr,
				      sizeof ( float ) * clusters,
				      cudaMemcpyDeviceToHost ) );
	
	  cluster_score = 0.0f;
	  for ( unsigned int i = 0; i < clusters; i++ )
	    {
	      cluster_score += host_medoids_costs_ptr [ i ];
	    }

	  if ( cluster_score < host_best_cluster_cost )
	    {
	      host_best_cluster_cost = cluster_score;
	      CUDA_SAFE_CALL( cudaMemcpy( dev_best_medoids_indexes_ptr,
					  dev_new_medoids_indexes_ptr,
					  sizeof ( unsigned int ) * clusters,
					  cudaMemcpyDeviceToDevice ) );
	    }

#ifdef _DEBUG
	  {
	    std::cout << "Clustering Score\n";
	    std::cout << cluster_score << "\n";
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

      if ( iterations < iterations_limit )
	{
	std::cout << "\rConverged after " <<
	  ( iterations + 1 ) << " iterations.";
	}
      else
	{
	std::cout << "\nReached iteration limit @ " <<
	  ( iterations ) << " / " << iterations_limit << "." << std::endl;
	}
    } while ( runs++ < repetitions );

  // Print Results
  // Restore assoc. table of best medoid set.
  associate_closest_medoid
    <<< 1, threads >>> ( clusters,
			 elements,
			 dev_best_medoids_indexes_ptr,
			 dev_medoid_assoc_ptr,
			 dev_distance_matrix_ptr );

  // Copy restored assoc. table.
  CUDA_SAFE_CALL( cudaMemcpy( host_medoid_assoc_ptr,
			      dev_medoid_assoc_ptr,
			      sizeof ( unsigned int ) * elements,
			      cudaMemcpyDeviceToHost ) );

  // Copy best medoid set indexes.
  CUDA_SAFE_CALL( cudaMemcpy( host_medoid_indexes_ptr,
			      dev_best_medoids_indexes_ptr,
			      sizeof ( unsigned int ) * clusters,
			      cudaMemcpyDeviceToHost ) );

  output << "Final Clustering Score: " << host_best_cluster_cost << "\n";

  unsigned int cluster_it = 0;
  while ( cluster_it < clusters )
    {
      output << "#Cluster " << cluster_it << std::endl;
      unsigned int element_it = 0;
      unsigned int medoid_it = host_medoid_indexes_ptr [ cluster_it ];
      while ( element_it < elements )
	{
	  if ( medoid_it == host_medoid_assoc_ptr [ element_it ] )
	    output << ( element_it + 1 ) << std::endl;
	  element_it++;
	}
      cluster_it++;
    }

  // Host Memory Deallocation
  free( host_candidates_costs_ptr );
  free( host_medoids_costs_ptr );
  free( host_medoid_assoc_ptr );
  free( host_medoid_indexes_ptr );

  // Device Memory Deallocation
  CUDA_SAFE_CALL( cudaFree ( dev_distance_matrix_ptr ) );
  CUDA_SAFE_CALL( cudaFree ( dev_medoid_candidates_cost_ptr ) );
  CUDA_SAFE_CALL( cudaFree ( dev_medoids_costs_ptr ) );
  CUDA_SAFE_CALL( cudaFree ( dev_new_medoids_indexes_ptr ) );
  CUDA_SAFE_CALL( cudaFree ( dev_best_medoids_indexes_ptr ) );
  CUDA_SAFE_CALL( cudaFree ( dev_medoid_assoc_ptr ) );
  CUDA_SAFE_CALL( cudaFree ( dev_diff_flag_ptr ) );

  return;
}
