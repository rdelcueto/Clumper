/**
 * @file clumper.cpp GPGPU K-Medoid Clustering Implementation.
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

void print_usage ()
{
  std::cout <<
    "Usage: [NUMBER OF CLUSTERS] [MATRIX COLUMN/ROW ELEMENTS] [INPUT MATRIX FILE]\n";
  return;
}

int parse_pdist_matrix ( std::ifstream &file,
			 const unsigned int matrix_size,
			 float matrix[] )
{
  unsigned int x_coord;
  unsigned int y_coord;
  float value;

  while ( file.good() )
    {
      file >> x_coord >> y_coord >> value;
      x_coord--; // 1-based-arrays
      y_coord--; // 1-based-arrays
      if ( x_coord < matrix_size && y_coord < matrix_size )
	{
	  matrix [ y_coord * matrix_size + x_coord ] = value;
	  matrix [ x_coord * matrix_size + y_coord ] = value;
	}
    }
  return 0;
}

void print_input_matrix ( const unsigned int matrix_size,
			  const float matrix[] )
{
  for ( unsigned int i = 0; i < matrix_size; i++ )
    {
      for ( unsigned int j = i + 1; j < matrix_size; j++ )
	{
	  std::cout << matrix [ i * matrix_size + j ];
	  if ( i != ( matrix_size - 2 ) || j != ( matrix_size - 1 ) )
	    std::cout << ", ";
	}
    }
  std::cout << ';' << std::endl;
}

int main ( const int argc, const char **argv )
{
  unsigned int k = 0;
  unsigned int input_matrix_width = 0;
  float *input_matrix;

  if ( argc != 4 )
    {
      std::cerr << "Error: Wrong argument format.\n";
      print_usage();
      return EXIT_FAILURE;
    }
  else
    {
      k = atoi ( argv [ 1 ] );
      input_matrix_width = atoi ( argv [ 2 ] );
      std::ifstream input_matrix_file ( argv [ 3 ] );

      if ( !input_matrix_file.is_open() )
	{
	  std::cerr << "IOError: Couldn't open input file.\n";
	  return EXIT_FAILURE;
	}
      else
	{
	  input_matrix = ( float* ) calloc ( input_matrix_width * input_matrix_width,
					     sizeof ( float ) );
	  parse_pdist_matrix( input_matrix_file, input_matrix_width, input_matrix );
	  input_matrix_file.close();
	  //print_input_matrix( input_matrix_width, input_matrix );

	  k_medoid_clustering( k, input_matrix, input_matrix_width, 5 );
          
	  free ( input_matrix );
	}
    }
  return EXIT_SUCCESS;
}
