/**
 * @file clumper_kernels.h GPGPU K-Medoid Clustering GPU Kernels Header
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

#ifndef __CLUMPER_KERNELS_H__
#define __CLUMPER_KERNELS_H__

#include <iostream>
#include <sstream>
#include <set>
#include <limits>

#include <math_constants.h>

/** 
 * Fills the medoid_costs vector with every medoids assigned cost, given the association
 * table.
 *
 * @param cluster Index of cluster to compute candidates.
 * @param elements Number of elements.
 * @param medoid_indexes Array with indexes describing the current medoid set.
 * @param medoid_assoc Array describing the medoid association table of all elements.
 * @param distance_matrix The input distance matrix.
 * @param medoids_costs Array for writing each medoid cost.
 */
__global__
void get_current_medoids_cost ( const unsigned int clusters,
				const unsigned int elements,
				const unsigned int medoid_indexes[],
				const unsigned int medoid_assoc[],
				const float distance_matrix[],
				float medoids_costs[] );

/** 
 * Fills up the medoid_assoc vector with index to the closest medoid from the current
 * medoid set, for every element.
 * 
 * @param clusters Number of clusters.
 * @param elements Number of elements.
 * @param medoid_indexes Array with indexes describing the current medoid set.
 * @param medoid_assoc Array describing the medoid association for every element.
 * @param distance_matrix The input distance matrix.
 */
__global__
void associate_closest_medoid ( const unsigned int clusters,
				const unsigned int elements,
				const unsigned int medoid_indexes[],
				unsigned int medoid_assoc[],
				const float distance_matrix[] );

/** 
 * Clears previous iteration candidates' cost values and blacklisted elements.
 *
 * @param elements Number of elements.
 * @param medoid_candidates_cost Array for clearing values.
 */
__global__
void clear_medoid_candidates ( const unsigned int elements,
			       float medoid_candidates_cost[] );

/** 
 * Computes the cost for all elements as candidates for substitution of medoid
 * for the given cluster.
 *
 * @param cluster_idx Index of cluster to compute candidates.
 * @param elements Number of elements.
 * @param medoid_assoc Array describing the medoid association table of all elements.
 * @param distance_matrix The input distance matrix.
 * @param medoid_candidates_cost Array for writing each candidate's cost.
 */
__global__
void compute_medoid_candidates ( const unsigned int cluster_idx,
				 const unsigned int elements,
				 const unsigned int medoid_assoc[],
				 const unsigned int medoid_indexes[],
				 const float distance_matrix[],
				 float medoid_candidates_cost[] );

/** 
 * Updates the medoid indexes n' costs with the best candidate if it exists.
 *
 * @param cluster_idx Index of cluster to reduce.
 * @param elements Number of elements.
 * @param medoid_assoc Array describing the medoid association table of all elements.
 * @param medoid_candidates_cost Array describing each candidate's cost.
 * @param medoid_indexes Array describing the current medoid set.
 * @param medoid_costs Array describing the current medoids' costs.
 * @param diff_flag Difference flag to indicate if there was a change in the medoid set.
 */
__global__
void reduce_medoid_candidates ( const unsigned int cluster_idx,
				const unsigned int elements,
				unsigned int medoid_assoc[],
				float medoid_candidates_cost[],
				unsigned int medoid_indexes[],
				float medoid_costs[],
				int *diff_flag );

#endif // __CLUMPER_KERNELS_H__
