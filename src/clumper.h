/**
 * @file clumpper.h GPGPU K-Medoid Clustering Header
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

#ifndef __CLUMPER_H__
#define __CLUMPER_H__

#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <set>
#include <limits>

#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <argtable2.h>

#include <cutil.h>
#include <math_constants.h>

/** 
 * Prints the program's usage.
 * 
 */
void print_usage ();

/** 
 * Will process text data, and import the computed the distance matrix.
 *
 * \attention Input format for every vector of dimension d, is as follows:
 * [v_0] [v_1] ... [v_d]
 * Every vector must be of the same dimension.
 *
 * @param file Reading input data.
 * @param vectors_dim Dimension of input vectors
 * @param vectors_size Number of vectors.
 * @param vectors Vectors array.
 * @param matrix Resulting symmetric distance matrix.
 *
 * @return returns error status.
 */
int create_pdist_matrix ( std::ifstream &file,
			  const unsigned int vectors_dim,
			  const unsigned int vectors_size,
			  float vectors[],
			  float matrix[] );

/** 
 * Imports text data into a symmetric matrix array.
 *
 * \attention Input format is as follows: element-row element-column value
 * Matrix array is initialized with zeros. 
 * Unspecified elements are left with a zero value.
 * Since the resulting matrix is symmetric, the [A] [B] [X] input, writes into the
 * (a,b) and (b,a) elements of the matrix vector.
 * 
 * @param file Reading input data matrix.
 * @param index_base Input format index base for 0-indexed or 1-indexed vectors.
 * If index_base != 0, then matrix is parsed with 1-indexed values.
 * @param matrix_size Integer that defines the square matrix dimension.
 * @param matrix Resulting symmetric distance matrix
 * 
 * @return returns error status.
 */
int parse_pdist_matrix ( std::ifstream &file,
			 const int index_base,
			 const int matrix_size,
			 float matrix[] );

/** 
 * Prints the stored square matrix.
 * 
 * @param matrix_size Integer that defines the square matrix dimension.
 * @param matrix Square matrix array.
 */
void print_input_matrix ( const unsigned int matrix_size,
			  const float matrix[] );

/** 
 * Initializes the K-medoid set, using a random selection of K random elements, from
 * the input data points.
 * 
 * @param k Number of medoids to initialize. 
 */
void init_k_cluster_medoids ( const unsigned k );

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
void reduce_medoid_candidates ( const unsigned int cluster_idx,
				const unsigned int elements,
				unsigned int medoid_assoc[],
				float medoid_candidates_cost[],
				unsigned int medoid_indexes[],
				float medoid_costs[],
				int *diff_flag );

/** 
 * Runs the K-Medoid Clustering Algorithm Function over the input data, and
 * returns the resulting medoids and their associated data-points.
 *
 * The function will return the best clustering set after the specified repetitions
 * 
 * @param clusters Number of clusters ( K parameter )
 * @param distance_matrix Input Distance Squared Matrix
 * @param elements Number of Elements / Distance Matrix Size.
 * @param max_iterations Maximum Number of Iterations. A zero value, disables limit.
 * @param repetitions Number of repetitions to run.
 * @param seed Pseudo Random Generator Seed.
 * If zero, current system time will be used as seed.
 * @param output Output stringstream object for writing results.
 */
void k_medoid_clustering ( const unsigned int clusters,
			   const float distance_matrix[],
			   const unsigned int elements,
			   const unsigned int max_iterations,
			   const unsigned int repetitions,
			   const unsigned long seed,
			   std::stringstream &output );
#endif // __CLUMPER_H__
