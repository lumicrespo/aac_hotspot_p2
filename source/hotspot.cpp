#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>
#include "time.h"
#include "kernel.h"

// Returns the current system time 
double get_time(){
    return ((double)clock())*1000/CLOCKS_PER_SEC;
}

using namespace std;

#define BLOCK_SIZE 16
#define BLOCK_SIZE_C BLOCK_SIZE
#define BLOCK_SIZE_R BLOCK_SIZE

#define STR_SIZE	256

/* maximum power density possible (say 300W for a 10mm x 10mm chip)	*/
#define MAX_PD	(3.0e6)
/* required precision in degrees	*/
#define PRECISION	0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor	*/
#define FACTOR_CHIP	0.5
#define OPEN
//#define NUM_THREAD 4


/* chip parameters	*/
const float t_chip = 0.0005;
const float chip_height = 0.016;
const float chip_width = 0.016;

/* ambient temperature, assuming no package at all	*/
const float amb_temp = 80.0;

int num_omp_threads;


/********************************** NEW ************************************/
/* variables to identify critical sections */

double total_time_ifs =0;
double total_time_loop=0;

double total_time_tran_temp=0;
double total_time_single_iteration=0;

/***************************************************************************/

/* Single iteration of the transient solver in the grid model.
 * advances the solution of the discretized difference equations 
 * by one time step
 */
void single_iteration(float *result, float *temp, float *power, int row, int col,
					  float Cap_1, float Rx_1, float Ry_1, float Rz_1, float step)
{
	double start_time_single_iteration=get_time();
    int r, c;
    int chunk;
    int num_chunk = row*col / (BLOCK_SIZE_R * BLOCK_SIZE_C);
    int chunks_in_row = col/BLOCK_SIZE_C;
    int chunks_in_col = row/BLOCK_SIZE_R;
	float *delta = (float *) calloc (1, sizeof(float));
    
    #if defined(NEON) || defined(SVE)
	    for ( chunk = 0; chunk < num_chunk; ++chunk )
	    {
            int r_start = BLOCK_SIZE_R*(chunk/chunks_in_col);
            int c_start = BLOCK_SIZE_C*(chunk%chunks_in_row); 
            int r_end = r_start + BLOCK_SIZE_R > row ? row : r_start + BLOCK_SIZE_R;
            int c_end = c_start + BLOCK_SIZE_C > col ? col : c_start + BLOCK_SIZE_C;
           
	       
            if ( r_start == 0 || c_start == 0 || r_end == row || c_end == col )
            {	
	    		double start_time_ifs = get_time();
	    		
                for ( r = r_start; r < r_start + BLOCK_SIZE_R; ++r ) 
	    		{
	    			kernel_ifs(result, temp, power, (size_t)c_start, (size_t)BLOCK_SIZE_C, (size_t)col, (size_t)r,(size_t) row, Cap_1, Rx_1, Ry_1, Rz_1, amb_temp, delta);
	    		}
                
                double end_time_ifs = get_time();
	    		total_time_ifs += (end_time_ifs - start_time_ifs);
	    		continue;
	    	}
        }
	    
	    double start_time_loop = get_time();
	    for ( r = BLOCK_SIZE_R; r < row - BLOCK_SIZE_R ; ++r ) {
	    	kernel(result, temp, power, (size_t)BLOCK_SIZE_C, (size_t)(col-BLOCK_SIZE_C), (size_t)col, (size_t)r, Cap_1, Rx_1, Ry_1, Rz_1, amb_temp);
	    }
	    double end_time_loop = get_time();
        total_time_loop +=(end_time_loop - start_time_loop);
	    
	    double start_time_ifs = get_time();
	    result[0] = temp[0]+ (Cap_1) * (power[0] +
	    			(temp[1] - temp[0]) * Rx_1 +
	    			(temp[col] - temp[0]) * Ry_1 +
	    			(amb_temp - temp[0]) * Rz_1);  

	    result[(row-1)*col+col-1] = temp[(row-1)*col+col-1] + (Cap_1) * (power[(row-1)*col+col-1] +
	    							(temp[(row-1)*col+col-2] - temp[(row-1)*col+col-1]) * Rx_1 +
	    							(temp[(row-2)*col+col-1] - temp[(row-1)*col+col-1]) * Ry_1 +
	    							(amb_temp - temp[(row-1)*col+col-1]) * Rz_1);
	       

	    result[(row-1)*col] =temp[(row-1)*col] + (Cap_1) * (power[(row-1)*col] +
	    					(temp[(row-1)*col+1] - temp[(row-1)*col]) * Rx_1 +
	    					(temp[(row-2)*col] - temp[(row-1)*col]) * Ry_1 +
	    					(amb_temp - temp[(row-1)*col]) * Rz_1);
        double end_time_ifs = get_time();
        total_time_ifs += (end_time_ifs - start_time_ifs);
	    					
	    double end_time_single_iteration= get_time();
	    total_time_single_iteration+= (end_time_single_iteration - start_time_single_iteration);

    #elif defined(ORIGINAL)
        //original code
        for ( chunk = 0; chunk < num_chunk; ++chunk )
	    {
            int r_start = BLOCK_SIZE_R*(chunk/chunks_in_col);
            int c_start = BLOCK_SIZE_C*(chunk%chunks_in_row); 
            int r_end = r_start + BLOCK_SIZE_R > row ? row : r_start + BLOCK_SIZE_R;
            int c_end = c_start + BLOCK_SIZE_C > col ? col : c_start + BLOCK_SIZE_C;
           
	       
            if ( r_start == 0 || c_start == 0 || r_end == row || c_end == col )
            {	
	    		double start_time_ifs = get_time();
	    		
                for ( r = r_start; r < r_start + BLOCK_SIZE_R; ++r ) 
	    		{
	    			kernel_ifs(result, temp, power, (size_t)c_start, (size_t)BLOCK_SIZE_C, (size_t)col, (size_t)r,(size_t) row, Cap_1, Rx_1, Ry_1, Rz_1, amb_temp, delta);
	    		}
                
                double end_time_ifs = get_time();
	    		total_time_ifs += (end_time_ifs - start_time_ifs);
	    		continue;
	    	}

            double start_time_loop = get_time();
            for ( r = BLOCK_SIZE_R; r < row - BLOCK_SIZE_R ; ++r ) {
                kernel(result, temp, power, (size_t)c_start, (size_t)(c_end), (size_t)col, (size_t)r, Cap_1, Rx_1, Ry_1, Rz_1, amb_temp);
            }
            double end_time_loop = get_time();
            total_time_loop +=(end_time_loop - start_time_loop);
        }
	    
        double end_time_single_iteration= get_time();
	    total_time_single_iteration+= (end_time_single_iteration - start_time_single_iteration);
 
    #else 
        for ( chunk = 0; chunk < num_chunk; ++chunk )
	    {
            int r_start = BLOCK_SIZE_R*(chunk/chunks_in_col);
            int c_start = BLOCK_SIZE_C*(chunk%chunks_in_row); 
            int r_end = r_start + BLOCK_SIZE_R > row ? row : r_start + BLOCK_SIZE_R;
            int c_end = c_start + BLOCK_SIZE_C > col ? col : c_start + BLOCK_SIZE_C;
           
	       
            if ( r_start == 0 || c_start == 0 || r_end == row || c_end == col )
            {	
	    		double start_time_ifs = get_time();
	    		
                for ( r = r_start; r < r_start + BLOCK_SIZE_R; ++r ) 
	    		{
	    			kernel_ifs(result, temp, power, (size_t)c_start, (size_t)BLOCK_SIZE_C, (size_t)col, (size_t)r,(size_t) row, Cap_1, Rx_1, Ry_1, Rz_1, amb_temp, delta);
	    		}
                
                double end_time_ifs = get_time();
	    		total_time_ifs += (end_time_ifs - start_time_ifs);
	    		continue;
	    	}
        }
	    
	    double start_time_loop = get_time();
	    for ( r = BLOCK_SIZE_R; r < row - BLOCK_SIZE_R ; ++r ) {
	    	kernel(result, temp, power, (size_t)BLOCK_SIZE_C, (size_t)(col-BLOCK_SIZE_C), (size_t)col, (size_t)r, Cap_1, Rx_1, Ry_1, Rz_1, amb_temp);
	    }
	    double end_time_loop = get_time();
        total_time_loop +=(end_time_loop - start_time_loop);
	    
	    double start_time_ifs = get_time();
	    result[0] = temp[0]+ (Cap_1) * (power[0] +
	    			(temp[1] - temp[0]) * Rx_1 +
	    			(temp[col] - temp[0]) * Ry_1 +
	    			(amb_temp - temp[0]) * Rz_1);  

	    result[(row-1)*col+col-1] = temp[(row-1)*col+col-1] + (Cap_1) * (power[(row-1)*col+col-1] +
	    							(temp[(row-1)*col+col-2] - temp[(row-1)*col+col-1]) * Rx_1 +
	    							(temp[(row-2)*col+col-1] - temp[(row-1)*col+col-1]) * Ry_1 +
	    							(amb_temp - temp[(row-1)*col+col-1]) * Rz_1);
	       

	    result[(row-1)*col] =temp[(row-1)*col] + (Cap_1) * (power[(row-1)*col] +
	    					(temp[(row-1)*col+1] - temp[(row-1)*col]) * Rx_1 +
	    					(temp[(row-2)*col] - temp[(row-1)*col]) * Ry_1 +
	    					(amb_temp - temp[(row-1)*col]) * Rz_1);
        double end_time_ifs = get_time();
        total_time_ifs += (end_time_ifs - start_time_ifs);
	    					
	    double end_time_single_iteration= get_time();
	    total_time_single_iteration+= (end_time_single_iteration - start_time_single_iteration);
    #endif

}

/* Transient solver driver routine: simply converts the heat 
 * transfer differential equations to difference equations 
 * and solves the difference equations by iterating
 */
void compute_tran_temp(float *result, int num_iterations, float *temp, float *power, int row, int col) 
{
	double start_time_tran_temp=get_time();
	#ifdef VERBOSE
	int i = 0;
	#endif

	float grid_height = chip_height / row;
	float grid_width = chip_width / col;

	float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
	float Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
	float Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
	float Rz = t_chip / (K_SI * grid_height * grid_width);

	float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    float step = PRECISION / max_slope / 1000.0;

    float Rx_1=1.f/Rx;
    float Ry_1=1.f/Ry;
    float Rz_1=1.f/Rz;
    float Cap_1 = step/Cap;
	#ifdef VERBOSE
	fprintf(stdout, "total iterations: %d s\tstep size: %g s\n", num_iterations, step);
	fprintf(stdout, "Rx: %g\tRy: %g\tRz: %g\tCap: %g\n", Rx, Ry, Rz, Cap);
	#endif

        int array_size = row*col;
        {
            float* r = result;
            float* t = temp;
            for (int i = 0; i < num_iterations ; i++)
            {
                #ifdef VERBOSE
                fprintf(stdout, "iteration %d\n", i++);
                #endif
                single_iteration(r, t, power, row, col, Cap_1, Rx_1, Ry_1, Rz_1, step);
                float* tmp = t;
                t = r;
                r = tmp;
            }	
        }
	#ifdef VERBOSE
	fprintf(stdout, "iteration %d\n", i++);
	#endif
	double end_time_tran_temp=get_time();
	total_time_tran_temp+= (end_time_tran_temp - start_time_tran_temp);
}

void fatal(const char *s)
{
	fprintf(stderr, "error: %s\n", s);
	exit(1);
}

void writeoutput(float *vect, int grid_rows, int grid_cols, char *file) {

    int i,j, index=0;
    FILE *fp;
    char str[STR_SIZE];

    if( (fp = fopen(file, "w" )) == 0 )
        printf( "The file was not opened\n" );


    for (i=0; i < grid_rows; i++) 
        for (j=0; j < grid_cols; j++)
        {

            sprintf(str, "%d\t%g\n", index, vect[i*grid_cols+j]);
            fputs(str,fp);
            index++;
        }

    fclose(fp);	
}

void read_input(float *vect, int grid_rows, int grid_cols, char *file)
{
  	int i, index;
	FILE *fp;
	char str[STR_SIZE];
	float val;

	fp = fopen (file, "r");
	if (!fp)
		fatal("file could not be opened for reading");

	for (i=0; i < grid_rows * grid_cols; i++) {
		fgets(str, STR_SIZE, fp);
		if (feof(fp))
			fatal("not enough lines in file");
		if ((sscanf(str, "%f", &val) != 1) )
			fatal("invalid file format");
		vect[i] = val;
	}

	fclose(fp);	
}

void usage(int argc, char **argv)
{
	fprintf(stderr, "Usage: %s <grid_rows> <grid_cols> <sim_time> <no. of threads><temp_file> <power_file>\n", argv[0]);
	fprintf(stderr, "\t<grid_rows>  - number of rows in the grid (positive integer)\n");
	fprintf(stderr, "\t<grid_cols>  - number of columns in the grid (positive integer)\n");
	fprintf(stderr, "\t<sim_time>   - number of iterations\n");
	fprintf(stderr, "\t<no. of threads>   - number of threads\n");
	fprintf(stderr, "\t<temp_file>  - name of the file containing the initial temperature values of each cell\n");
	fprintf(stderr, "\t<power_file> - name of the file containing the dissipated power values of each cell\n");
        fprintf(stderr, "\t<output_file> - name of the output file\n");
	exit(1);
}

int main(int argc, char **argv)
{
	double start_time = get_time();
	int grid_rows, grid_cols, sim_time, i;
	float *temp, *power, *result;
	char *tfile, *pfile, *ofile;
	
	/* check validity of inputs	*/
	if (argc != 8)
		usage(argc, argv);
	if ((grid_rows = atoi(argv[1])) <= 0 ||
		(grid_cols = atoi(argv[2])) <= 0 ||
		(sim_time = atoi(argv[3])) <= 0 || 
		(num_omp_threads = atoi(argv[4])) <= 0
		)
		usage(argc, argv);
	
	double start_time_init = get_time();
	
	/* allocate memory for the temperature and power arrays	*/
	temp = (float *) calloc (grid_rows * grid_cols, sizeof(float));
	power = (float *) calloc (grid_rows * grid_cols, sizeof(float));
	result = (float *) calloc (grid_rows * grid_cols, sizeof(float));

	if(!temp || !power)
		fatal("unable to allocate memory");

	/* read initial temperatures and input power	*/
	pfile = argv[6];
    ofile = argv[7];
	tfile = argv[5];

	read_input(temp, grid_rows, grid_cols, tfile);
	read_input(power, grid_rows, grid_cols, pfile);
	
	double end_time_init = get_time();
	
	printf("Start computing the transient temperature\n");
	
    

    compute_tran_temp(result,sim_time, temp, power, grid_rows, grid_cols);

    
	double start_output_time_init = get_time();
    writeoutput((1&sim_time) ? result : temp, grid_rows, grid_cols, ofile);
	double end_output_time_init = get_time();



	/* output results	*/
#ifdef VERBOSE
	fprintf(stdout, "Final Temperatures:\n");
#endif

#ifdef OUTPUT
	for(i=0; i < grid_rows * grid_cols; i++)
	fprintf(stdout, "%d\t%g\n", i, temp[i]);
#endif
	/* cleanup	*/
	free(temp);
	free(power);
	
	double end_time = get_time();

    printf("Ending simulation\n");
    printf("Total time: %lf\n", (end_time - start_time));
	
	printf("Total time in initialization: %lf\n", (end_time_init - start_time_init)+(end_output_time_init-start_output_time_init));
	
    printf("Total time in compute_tran_temp: %lf\n", total_time_tran_temp);
	
	printf("Total time in single_iteration: %lf\n", total_time_single_iteration);
	
    printf("Total time in ifs loop: %lf\n", total_time_ifs);
    printf("Total time in loop: %lf\n", total_time_loop);

	return 0;
}
/* vim: set ts=4 sw=4  sts=4 et si ai: */
