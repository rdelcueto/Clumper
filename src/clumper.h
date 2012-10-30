/**
 * @file clumper.h GPGPU K-Medoid Clustering GPU Kernels Header
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

#include <string.h>
#include <sys/time.h>
#include <argtable2.h>

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
