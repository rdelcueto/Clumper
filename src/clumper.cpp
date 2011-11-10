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

// Argument Table Variables
struct arg_lit *help, *version;
struct arg_str *out_file_prefix, *rnd_seed, *k_list;
struct arg_int *k_range_init, *k_range_end, *k_range_step, *max_iter,
  *n_rep, *n_elements, *dimension, *idx_base;
struct arg_file *k_list_file, *in_pdist_matrix_file, *in_vector_file;
struct arg_rem *input_options, *k_options, *cluster_options;
struct arg_end *end;

int create_pdist_matrix ( std::ifstream &file,
			  const unsigned int vectors_dim,
			  const unsigned int vectors_size,
			  float vectors[],
			  float matrix[] )
{
  // Parse input vectors
  unsigned int vector_count = 0;
  unsigned int dim_count = 0;
  float value;
  while ( file.good() )
    {
      file >> value;
      vectors [ vector_count * vectors_dim + dim_count++ ] = value;

      if ( ( dim_count % vectors_dim ) == 0 )
	{
	  vector_count++;
	  dim_count = 0;
	}
    }

  // Compute Pdist-Matrix
  for ( unsigned int i = 0; i < vectors_size; i++ )
    {
      float local_vector_cache[ vectors_dim ];
      memcpy( &local_vector_cache,
	      &vectors [ vectors_dim * i ],
	      sizeof ( float ) * vectors_dim );

      for ( unsigned int j = 0; j < vectors_size; j++ )
	{
	  float accum_distance = 0;
	  float component_distance;
	  for ( unsigned int k = 0; k < vectors_dim; k++ )
	    {
	      component_distance =
		local_vector_cache [ k ] -
		vectors [ j * vectors_dim + k ];
	      accum_distance += component_distance * component_distance;
	    }
	  if ( i == j )
	    {
	      matrix [ i * vectors_size + j ] =
		sqrtf( accum_distance );
	    }
	  else
	    {
	      matrix [ j * vectors_size + i ] = 
		matrix [ i * vectors_size + j ] =
		sqrtf( accum_distance );
	    }
	}
    }
  return 0;
}

int parse_pdist_matrix ( std::ifstream &file,
			 const int index_base,
			 const unsigned int matrix_size,
			 float matrix[] )
{
  unsigned int x_coord;
  unsigned int y_coord;
  float value;

  while ( file.good() )
    {
      file >> x_coord >> y_coord >> value;
      if ( index_base )
	{
	  x_coord--; // 1-based-arrays
	  y_coord--; // 1-based-arrays
	}
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

int main ( int argc, char **argv )
{
  const char *progname = "clumper";

  // Argtable Setup
  void *argtable[] = {
    help = arg_lit0( NULL, "help", "display this help and exit" ),
    version = arg_lit0( NULL, "version", "display version information and exit" ),
    input_options = arg_rem ( "\nINPUT OPTIONS", "\n\nTo execute Clumper, the user MUST specify the number of elements the program must read from the input data files.\nWhen using a vector array as input (see --input-vector option), the user must specify the dimension of this vectors. (see --vector-dimension options). When using a distance matrix as input (see --pdist-matrix), the user may specify the index-base of the input elements coordinates.\n" ),
    n_elements = arg_int1( "n", "elements", "<int>",
			   "specify number of elements/entries to read\n" ),

    in_vector_file = arg_file0( "v", "input-vector", "<file>",
				"specify input vector file" ),
    dimension = arg_int0( "d", "vector-dimension", "<int>",
			  "specify input vectors dimensions\n" ),

    in_pdist_matrix_file = arg_file0( "p", "pdist-matrix", "<file>",
				      "specify distance matrix file" ),
    idx_base = arg_int0( "b", "index-base", "<0|1>",
			 "define the matrix input index base. Default = 0" ),

    k_options = arg_rem ( "\nK OPTIONS", "\n\nTo execute Clumper, the user must also specify in one way, the K argument value(s) to compute. Clumper accepts three variants.\n1)Integer list file: A file containing a list of the k values computed.\n2)String listing: A string with the k values\n3)K-Range: A set of 3 arguments, specifying Initial K, Terminal K and a K-step value.(see --k-range-init, --k-range-end, --k-range-step options)\n" ),

    k_list_file = arg_file0( NULL, "k-list-file", "<file>", "specify k list file" ),
    k_list = arg_str0( "l", "k-list", "<string>", "specify k list" ),
    k_range_init = arg_int0( "i", "k-range-init", "<int>",
			     "specify initial k-range value (inclusive)" ),
    k_range_end = arg_int0( "e", "k-range-end", "<int>",
			    "specify terminal k-range value (exclusive)" ),
    k_range_step = arg_int0( "s", "k-range-step", "<int>",
			     "specify k-range stepping value" ),

    cluster_options = arg_rem ("\nCLUSTERING OPTIONS", "\n\nThe user may specify the following additional options for the clustering algorithm execution.\n" ),

    rnd_seed = arg_str0( NULL, "seed", "<long long>",
			 "specify the pseudo-random generator seed" ),
    max_iter = arg_int0( "m", "max-iterations", "<int>",
			 "define maximum number of iterations. Default = 10" ),
    n_rep = arg_int0( "r", "repetitions", "<int>",
		      "define number of repetitions per k. Default = 0" ),
    out_file_prefix = arg_str1( "o", "output-prefix", "<string>",
				"prefix to append to each output result file" ),
    end = arg_end(20),
  };

  // Initialize Argument Values
  rnd_seed->sval[ 0 ] = "0";
  max_iter->ival[ 0 ] = 10;
  n_rep->ival[ 0 ] = 0;
  idx_base->ival[ 0 ] = 0;

  // Argtable Parsing
  int nerrors = arg_parse( argc, argv, argtable);

  if ( help->count > 0 || argc < 2 )
    {
      std::cout << "Usage: " << progname << " -n <int> <INPUT OPTIONS> <K OPTIONS> [ EXTRA OPTIONS]\n\n";
      arg_print_glossary ( stdout, argtable, " %-25s %s\n" );
      std::cout << std::endl;
      return EXIT_SUCCESS;
    }

  if ( nerrors )
    {
      arg_print_errors( stdout, end, progname );
    }

  if ( version->count > 0 )
    {
      std::cout <<
	"\nClumper v0.9:\n" <<
	" Copyright (C) 2011 Rodrigo González del Cueto\n" <<
	" License GPLv2: GNU GPL version 2 <http://www.gnu.org/licenses/gpl-2.0.html>.\n" <<
	" This is free software; see the source for copying conditions.  There is NO\n" <<
	" warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n\n";
      return EXIT_SUCCESS;
    }

  if ( !in_pdist_matrix_file->count && !in_vector_file->count )
    {
      std::cerr << progname << ": invalid argument options. You must specify, ONE AND only ONE, of the following options, pdist-matrix | input-vector\n";
      nerrors++;
    }

  if ( !k_list_file->count && !k_list->count && !k_range_init->count )
    {
      std::cerr << progname << ": must specify one form of k parameter via k-list-file, k-list or k-range\n";
      nerrors++;
    }

  if ( ( k_list->count && k_range_init->count ) ||
       ( k_list->count && k_list_file->count ) ||
       ( k_list_file->count && k_range_init->count ) )
    {
      std::cerr << progname << ": only one of the following arguments can be specified... k-list-file | k-list | k-range arguments\n";
      nerrors++;
    }

  if ( nerrors )
    {
      return EXIT_FAILURE;
    }

  // Import Data Files
  std::ifstream input_file;
  std::string filename;
  unsigned int input_elements = n_elements->ival[ 0 ];
  float *input_matrix = ( float* ) calloc ( input_elements * input_elements,
					    sizeof ( float ) );

  if ( in_vector_file->count )
    {
      filename = in_vector_file->filename[ 0 ];
    }
  else
    {
      filename = in_pdist_matrix_file->filename[ 0 ];
    }

  input_file.open( filename.c_str() );
  if ( !input_file.is_open() )
    {
      std::cerr <<
	progname <<
	": IOError! Couldn't open input file... " <<
	filename << std::endl;
      return EXIT_FAILURE;
    }
  
  if ( in_vector_file->count )
    {
      if ( dimension->count )
	{
	  const unsigned int vectors_dim = dimension->ival[ 0 ];
	  float *vectors = ( float* ) calloc ( input_elements * vectors_dim,
					   sizeof ( float ) );

	  create_pdist_matrix ( input_file,
				vectors_dim,
				input_elements,
				vectors,
				input_matrix );
	}
      else
	{
	  std::cerr << progname <<
	    ": must specify vectors dimension, when using input-vector\n";
	  return EXIT_FAILURE;
	}
    }
  else
    {
      const int index_base = idx_base->ival[ 0 ];
      parse_pdist_matrix( input_file, index_base, input_elements, input_matrix );
    }
  input_file.close();
  //print_input_matrix( input_elements, input_matrix );

  // Static Argument Values
  std::string seed_string ( rnd_seed->sval[ 0 ] );
  const unsigned long seed = std::strtoul ( seed_string.c_str(), NULL, 10 );
  const unsigned int iterations = max_iter->ival[ 0 ];
  const unsigned int repetitions = n_rep->ival[ 0 ];
  std::string out_user_prefix ( out_file_prefix->sval[ 0 ] );
  std::stringstream output_string( std::stringstream::in | std::stringstream::out );
  std::stringstream output_filename( std::stringstream::in | std::stringstream::out );
  
  // Run over k-range
  if ( k_range_init->count )
    {
      if ( !k_range_end->count )
	{
	  std::cerr << progname << ": invalid k-range\n";
	  return EXIT_FAILURE;
	}

      const unsigned int init_k = k_range_init->ival[ 0 ];
      const unsigned int end_k = k_range_end->ival[ 0 ];
      unsigned int range_step = 1;

      if ( k_range_step->count )
	range_step = k_range_step->ival[ 0 ];
      
      if ( init_k >= end_k )
	{
	  std::cerr << progname << ": empty k range!\n";
	  return EXIT_SUCCESS;
	}
      for ( unsigned int k = init_k; k < end_k; k += range_step )
	{
	  if ( k > input_elements )
	    {
	      std::cerr <<
		progname <<
		": k value exceeds number of elements... skipping\n";
	    }
	  else
	    {
	      output_filename.str( std::string() );
	      output_filename << out_user_prefix << "_k" << k << ".txt";

	      output_string.str( std::string() );
	      std::cout << "Computing clustering for k = " << k << std::endl;

	      k_medoid_clustering( k,
				   input_matrix,
				   input_elements,
				   iterations,
				   repetitions,
				   seed,
				   output_string );
	      std::ofstream output_file;
	      output_file.open( output_filename.str().c_str() );
	      output_file << output_string.str();
	      output_file.close();
	    }
	}
    }
  else if ( k_list->count )
    {
      unsigned int k;

      std::istringstream tokenizer_stream ( k_list->sval[ 0 ] );
      std::string k_token;

      while ( getline( tokenizer_stream, k_token, ' ' ) )
	{
	  k = std::strtoul ( k_token.c_str(), NULL, 10 );

	  if ( k > input_elements )
	    {
	      std::cerr <<
		progname <<
		": k value exceeds number of elements... skipping\n";
	    }
	  else
	    {
	      output_filename.str( std::string() );
	      output_filename << out_user_prefix << "_k" << k << ".txt";

	      output_string.str( std::string() );
	      std::cout << "Computing clustering for k = " << k << std::endl;

	      k_medoid_clustering( k,
				   input_matrix,
				   input_elements,
				   iterations,
				   repetitions,
				   seed,
				   output_string );
	      std::ofstream output_file;
	      output_file.open( output_filename.str().c_str() );
	      output_file << output_string.str();
	      output_file.close();
	    }
	}
    }
  else if ( k_list_file->count )
    {
      std::string k_list_filename ( k_list_file->filename[ 0 ] );
      std::ifstream input_list_file;
      input_list_file.open ( k_list_filename.c_str() );
      if ( !input_list_file.is_open() )
	{
	  std::cerr <<
	    progname <<
	    ": IOError! Couldn't open k-list file..." <<
	    k_list_filename << std::endl;
	  return EXIT_FAILURE;
	}

      unsigned int k;
      while ( input_list_file.good() )
	{
	  input_list_file >> k;

	  if ( k > input_elements )
	    {
	      std::cerr <<
		progname <<
		": k value exceeds number of elements... skipping\n";
	    }
	  else
	    {
	      output_filename.str( std::string() );
	      output_filename << out_user_prefix << "_k" << k << ".txt";

	      output_string.str( std::string() );
	      std::cout << "Computing clustering for k = " << k << std::endl;

	      k_medoid_clustering( k,
				   input_matrix,
				   input_elements,
				   iterations,
				   repetitions,
				   seed,
				   output_string );
	      std::ofstream output_file;
	      output_file.open( output_filename.str().c_str() );
	      output_file << output_string.str();
	      output_file.close();
	    }

	}
    }

  free ( input_matrix );
  arg_freetable ( argtable, sizeof( argtable ) / sizeof( argtable [ 0 ] ));
  return EXIT_SUCCESS;
}
