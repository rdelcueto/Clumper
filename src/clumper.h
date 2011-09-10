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

#include <cmath>
#include <stdlib.h>
#include <algorithm>
#include <set>
#include <limits>

#include <cutil.h>
#include <math_constants.h>

/** 
 * Prints the program's usage.
 * 
 */
void print_usage ();

/** 
 * Imports text data into a symmetric matrix array.
 * Text input per line format: [X coordinate] [Y coordinate] [Value]
 *
 * \attention Matrix array is initialized with zeros. 
 * Unspecified elements are left with a zero value.
 * Since the resulting matrix is symmetric, the [A] [B] [X] input, writes into the
 * (a,b) and (b,a) elements of the matrix vector.
 * 
 * @param file Reading input data.
 * @param matrix_size Integer that defines the square matrix dimension.
 * @param matrix Storage array for the imported matrix data.
 * 
 * @return 
 */
int parse_input_matrix ( std::ifstream &file,
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
 * Fills up the medoid_assoc vector with index to the closest medoid from the current
 * medoid set, for every element.
 * 
 * @param clusters Number of clusters.
 * @param elements Number of elements.
 * @param medoid_indexes Array with indexes describing the selected medoid set.
 * @param medoid_assoc Array describing the medoid association for every element.
 * @param distance_matrix The input distance matrix.
 */
void associate_closest_medoid ( const unsigned int clusters,
				const unsigned int elements,
				const unsigned int medoid_indexes[],
				unsigned int medoid_assoc[],
				const float distance_matrix[] );

/** 
 * Fills up the medoid_candidates_cost vector, with the cost of every element's
 * selection as medoid within it's associated cluster and data-points.
 * 
 * @param clusters Number of clusters.
 * @param elements Number of elements.
 * @param medoid_assoc Array describing the medoid association for every element.
 * @param medoid_candidates_cost Array describing each candidate's cost.
 * @param distance_matrix The input distance matrix.
 */
void get_medoid_candidates_cost ( const unsigned int clusters,
				     const unsigned int elements,
				     const unsigned int medoid_assoc[],
				     float medoid_candidates_cost[],
				     const float distance_matrix[] );

/** 
 * Gets the best possible medoid for every cluster, writing into medoid_indexes 
 * given selection, and the cost of each selected medoid into medoids_cost.
 * 
 * @param clusters Number of clusters.
 * @param elements Number of elements.
 * @param medoid_indexes Array with indexes describing the selected medoid set.
 * @param medoid_assoc Array describing the medoid association for every element.
 * @param medoids_cost Array describing the final medoid cost.
 * @param medoid_candidates_cost Array describing each candidate's cost.
 */
void get_best_medoid_candidate ( const unsigned int clusters,
				 const unsigned int elements,
				 unsigned int medoid_indexes[],
				 const unsigned int medoid_assoc[],
				 float medoids_cost[],
				 const float medoid_candidates_cost[] );

/** 
 * Runs the K-Medoid Clustering Algorithm Function over the input data, and
 * returns the resulting medoids and their associated data-points.
 * 
 * @param clusters Number of clusters
 * @param distance_matrix Input Distance Squared Matrix
 * @param elements Number of Elements / Distance Matrix Size.
 */
void k_medoid_clustering ( const unsigned int clusters,
			   const float distance_matrix[],
			   const unsigned int elements );

#endif // __CLUMPER_H__
