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

__global__ void get_medoid_candidates_cost ( const unsigned int clusters,
					     const unsigned int elements,
					     const unsigned int medoid_assoc[],
					     float medoid_candidates_cost[],
					     const float distance_matrix[] )
{
  unsigned int id = threadIdx.x;
  while ( id < elements )
    {
      unsigned int associated_medoid = medoid_assoc [ id ];
      float candidate_medoid_cost = 0;
      
      for ( unsigned int i = 0; i < elements; i++ )
	{
	  if ( medoid_assoc [ i ] == associated_medoid )
	    {
	      // TODO: Check Cost function.
	      float d = distance_matrix [ elements * i + id ];
	      candidate_medoid_cost += d * d;
	    }
	}
      medoid_candidates_cost [ id ] = candidate_medoid_cost;
      id += blockDim.x;
    }
}

__global__ void get_best_medoid_candidate ( const unsigned int clusters,
					   const unsigned int elements,
					   unsigned int medoid_indexes[],
					   const unsigned int medoid_assoc[],
					   float medoids_cost[],
					   const float medoid_candidates_cost[] )
{
  float * medoid_cost_local_reduction =
    ( float * ) shared_memory;
  unsigned int * medoid_index_local_reduction =
    ( unsigned int * ) &medoid_cost_local_reduction [ elements ];

  for ( unsigned int curr_cluster = 0; curr_cluster < clusters; curr_cluster++ )
    {
      unsigned int curr_cluster_index = medoid_indexes [ curr_cluster ];

      // Over-block Reduction ( Device Max-threads < elements )
      {
	float a, b;
	unsigned int best_idx;

	a = medoid_assoc [ threadIdx.x ] == curr_cluster_index ?
	  medoid_candidates_cost [ threadIdx.x ] : CUDART_INF_F;
	best_idx = threadIdx.x;

	unsigned int id = threadIdx.x + gridDim.x;
	while ( id < elements )
	  {
	    b = medoid_assoc [ id ] == curr_cluster_index ?
	      medoid_candidates_cost [ id ] : CUDART_INF_F;

	    if ( b < a )
	      {
		a = b;
		best_idx = id;
	      }
	    id += gridDim.x;
	  }

	medoid_cost_local_reduction [ threadIdx.x ] = a;
	medoid_index_local_reduction [ threadIdx.x ] = best_idx;
      }
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
	    medoids_cost [ curr_cluster ] = medoid_cost_local_reduction [ 0 ];
	    medoid_indexes [ curr_cluster ] = medoid_index_local_reduction [ 0 ];
	  }
      }
    }
}

void k_medoid_clustering ( const unsigned int clusters,
			   const float distance_matrix[],
			   const unsigned int elements )
{
  // Random Seed Initialization
  srand48( time( NULL ) );

  std::set<unsigned int> old_medoid_indexes;
  std::set<unsigned int> new_medoid_indexes;
  std::set<unsigned int> medoid_indexes_diff;

  // Get Max Threads per Block on Default Device
  cudaDeviceProp prop;
  CUDA_SAFE_CALL ( cudaGetDeviceProperties( &prop, 0 ) );
  const unsigned int threads = elements <= prop.maxThreadsPerBlock ?
    elements : prop.maxThreadsPerBlock;

  std::cout << "Using " << threads << " threads.\n";
  
  // Initialize with Random Medoids
  while ( old_medoid_indexes.size() < clusters )
    {
      unsigned int rand_medoid =
	(unsigned int) ( ceil( drand48() * elements ) - 1 );

      old_medoid_indexes.insert( rand_medoid );
    }

  // Device Memory Pointers
  float *dev_distance_matrix_ptr; // Distance Matrix
  float *dev_medoid_candidates_cost_ptr; // Every Possible Medoid cost
  float *dev_medoids_cost_ptr; // Selected Medoids' cost
  unsigned int *dev_medoid_indexes_ptr; // Selected Medoids' Indexes
  unsigned int *dev_medoid_assoc_ptr; // Medoid association table


  // Device Memory Allocation
  CUDA_SAFE_CALL( cudaMalloc( ( void ** ) &dev_distance_matrix_ptr,
			      sizeof ( float ) * elements * elements ) );

  CUDA_SAFE_CALL( cudaMalloc( ( void ** ) &dev_medoid_candidates_cost_ptr,
			      sizeof ( float ) * elements ) );

  CUDA_SAFE_CALL( cudaMalloc( ( void ** ) &dev_medoids_cost_ptr,
			      sizeof ( float ) * clusters ) );

  CUDA_SAFE_CALL( cudaMalloc( ( void ** ) &dev_medoid_indexes_ptr,
			      sizeof ( unsigned int ) * clusters ) );
  
  CUDA_SAFE_CALL( cudaMalloc( ( void ** ) &dev_medoid_assoc_ptr,
			      sizeof ( unsigned int ) * elements ) );
  
  // Host to Device Data Initialization
  CUDA_SAFE_CALL( cudaMemcpy ( dev_distance_matrix_ptr,
			       distance_matrix,
			       sizeof ( float ) * elements * elements,
			       cudaMemcpyHostToDevice ) );
  
  // Medoid Set Copy to Device
  {
    unsigned int host_medoid_indexes [ clusters ];
    unsigned int i = 0;
    for ( std::set<unsigned int>::iterator it = old_medoid_indexes.begin();
	  it != old_medoid_indexes.end();
	  ++it )
      {
	host_medoid_indexes [ i++ ] = *it;
      }

    CUDA_SAFE_CALL( cudaMemcpy ( dev_medoid_indexes_ptr,
				 &host_medoid_indexes,
				 sizeof ( unsigned int ) * clusters,
				 cudaMemcpyHostToDevice ) );
  }

  // Check Medoid Set Copy
  {
    unsigned int host_medoid_indexes [ clusters ];
    CUDA_SAFE_CALL( cudaMemcpy( &host_medoid_indexes,
				dev_medoid_indexes_ptr,
				sizeof ( unsigned int ) * clusters,
				cudaMemcpyDeviceToHost ) );

#ifdef _DEBUG
    std::cout << "Device Medoid Indexes\n";
    for ( unsigned int i = 0; i < clusters; i++ )
      {
    	// Output Device Medoid Indexes.
    	std::cout << host_medoid_indexes [ i ] << "\n";
      }
#endif
  }


  // K Medoid Clustering Loop
  unsigned int iterations = 0;
  unsigned int unmodified_iterations = 0;
  do
    {
      std::cout << "Running Iteration: " << ( iterations + 1 ) << std::endl;

      // Associate each datapoint with closest medoid
      associate_closest_medoid
	<<< 1, threads >>> ( clusters,
			     elements,
			     dev_medoid_indexes_ptr,
			     dev_medoid_assoc_ptr,
			     dev_distance_matrix_ptr );

      {
	unsigned int host_medoid_assoc [ elements ];
	CUDA_SAFE_CALL( cudaMemcpy( &host_medoid_assoc,
				    dev_medoid_assoc_ptr,
				    sizeof ( unsigned int ) * elements,
				    cudaMemcpyDeviceToHost ) );

#ifdef _DEBUG
	{
	  // Check Device Medoid Assoc. DEBUG
	  std::cout << "Medoid Assoc\n";
	  for ( unsigned int i = 0; i < elements; i++ )
	    {
	      std::cout << host_medoid_assoc [ i ] << "\n";
	    }
	}
#endif
      }

      // Get new medoid candidates' cost
      get_medoid_candidates_cost
	<<< 1, threads >>> ( clusters,
			     elements,
			     dev_medoid_assoc_ptr,
			     dev_medoid_candidates_cost_ptr,
			     dev_distance_matrix_ptr );

      // Check Device Medoid cost. DEBUG
      {
	float host_medoids_candidates_cost [ elements ];
	CUDA_SAFE_CALL( cudaMemcpy( &host_medoids_candidates_cost,
				    dev_medoid_candidates_cost_ptr,
				    sizeof ( float ) * elements,
				    cudaMemcpyDeviceToHost ) );

#ifdef _DEBUG
	{
	  std::cout << "Medoid cost\n";
	  for ( unsigned int i = 0; i < elements; i++ )
	    {
	      std::cout << host_medoids_candidates_cost [ i ] << "\n";
	    }
	}
#endif
      }

      // Get Best Medoid for each cluster.
      get_best_medoid_candidate
	<<< 1, threads,	( sizeof ( float ) + sizeof ( unsigned int ) ) * elements >>>
	( clusters,
	  elements,
	  dev_medoid_indexes_ptr,
	  dev_medoid_assoc_ptr,
	  dev_medoids_cost_ptr,
	  dev_medoid_candidates_cost_ptr );

      // Copy New Medoid Set into Host
      unsigned int host_medoid_indexes [ clusters ];
      CUDA_SAFE_CALL( cudaMemcpy( &host_medoid_indexes,
				  dev_medoid_indexes_ptr,
				  sizeof ( unsigned int ) * clusters,
				  cudaMemcpyDeviceToHost ) );

      for ( unsigned int i = 0; i < clusters; i++ )
	{
#ifdef _DEBUG
	  {
	    // Output Device Medoid Indexes.
	    std::cout << host_medoid_indexes [ i ] << "\n";
	  }
#endif
	  new_medoid_indexes.insert( host_medoid_indexes [ i ] );
	}

      float host_medoids_cost [ clusters ];
      CUDA_SAFE_CALL( cudaMemcpy( &host_medoids_cost,
				  dev_medoids_cost_ptr,
				  sizeof ( float ) * clusters,
				  cudaMemcpyDeviceToHost ) );

      double clustering_score = 0.0;	
      for ( unsigned int i = 0; i < clusters; i++ )
	{
	  clustering_score += host_medoids_cost [ i ];
	}
      std::cout << "Clustering Score = " << clustering_score << std::endl;

#ifdef _DEBUG
      {
	// Check Device Medoid cost. DEBUG
	for ( unsigned int i = 0; i < clusters; i++ )
	  {
	    std::cout << host_medoids_cost [ i ] << "\n";
	  }
      }
#endif

#ifdef _DEBUG
      {
	//Check Medoid Sets. DEBUG 
	std::cout << "Old Medoid Set\n";
	for ( std::set<unsigned int>::iterator it = old_medoid_indexes.begin();
	      it != old_medoid_indexes.end();
	      ++it )
	  {
	    std::cout << *it << std::endl;
	  }
	std::cout << "New Medoid Set\n";
	for ( std::set<unsigned int>::iterator it = new_medoid_indexes.begin();
	      it != new_medoid_indexes.end();
	      ++it )
	  {
	    std::cout << *it << std::endl;
	  }
      }
#endif

      // Check if medoid set changed
      medoid_indexes_diff.clear();
      std::set_difference( old_medoid_indexes.begin(),
			   old_medoid_indexes.end(),
			   new_medoid_indexes.begin(),
			   new_medoid_indexes.end(),
			   std::inserter( medoid_indexes_diff,
					  medoid_indexes_diff.end()));

#ifdef _DEBUG
      {
	std::cout << "Diff Medoid Set\n";
	for ( std::set<unsigned int>::iterator it = medoid_indexes_diff.begin();
	      it != medoid_indexes_diff.end();
	      ++it )
	  {
	    std::cout << *it << std::endl;
	  }
      }
#endif

      if ( ( (int) medoid_indexes_diff.size() ) == 0 )
	unmodified_iterations++;
      else 
	unmodified_iterations = 0;

      // Clear & Swap Medoid Sets    
      old_medoid_indexes.clear();
      old_medoid_indexes.swap ( new_medoid_indexes );
      new_medoid_indexes.clear();

    } while ( ( unmodified_iterations < 2 ) && ( ++iterations < 10 ) );

  // Print Results
  // TODO: Standarize input/output format
  std::cout << "Computed " << iterations << " with " << unmodified_iterations <<
    " unmodified iterations.\n";

  std::cout << "Resulting Medoid Set\n";

  unsigned int host_medoid_assoc [ elements ];
  CUDA_SAFE_CALL( cudaMemcpy( &host_medoid_assoc,
			      dev_medoid_assoc_ptr,
			      sizeof ( unsigned int ) * elements,
			      cudaMemcpyDeviceToHost ) );

  unsigned int n = 0;
  for ( std::set<unsigned int>::iterator it = old_medoid_indexes.begin();
	it != old_medoid_indexes.end();
	++it )
    {
      std::cout << "#Cluster\t" << ++n << std::endl;
      unsigned int medoid_index = *it;
      for ( unsigned int i = 0; i < elements; i++ )
	{
	  if ( medoid_index == host_medoid_assoc [ i ] )
	    std::cout << ( i + 1 ) << "\n";
	}
    }

  // Host Memory Deallocation
  old_medoid_indexes.clear();
  new_medoid_indexes.clear();
  medoid_indexes_diff.clear();

  // Device Memory Deallocation
  CUDA_SAFE_CALL( cudaFree ( dev_distance_matrix_ptr ) );
  CUDA_SAFE_CALL( cudaFree ( dev_medoid_candidates_cost_ptr ) );
  CUDA_SAFE_CALL( cudaFree ( dev_medoids_cost_ptr ) );
  CUDA_SAFE_CALL( cudaFree ( dev_medoid_indexes_ptr ) );
  CUDA_SAFE_CALL( cudaFree ( dev_medoid_assoc_ptr ) );

  return;
}
