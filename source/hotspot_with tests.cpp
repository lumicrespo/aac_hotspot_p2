#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>
#include "time.h"
#include "kernel.h"

// Returns the current system time 
double get_time(){
    return ((double)clock())*10000000/CLOCKS_PER_SEC;
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

/***************************************************************************/

/* Single iteration of the transient solver in the grid model.
 * advances the solution of the discretized difference equations 
 * by one time step
 */
void single_iteration(float *result, float *temp, float *power, int row, int col,
					  float Cap_1, float Rx_1, float Ry_1, float Rz_1, float step)
{
    int r, c;
    int chunk;
    int num_chunk = row*col / (BLOCK_SIZE_R * BLOCK_SIZE_C);
    int chunks_in_row = col/BLOCK_SIZE_C;
    int chunks_in_col = row/BLOCK_SIZE_R;
	float delta;
    //float *teste = (float *) calloc (1024 * 1024, sizeof(float));

/*
    for ( r = 0; r < row; ++r ) 
    {
        //if (c_start == 0) 
        {
        result[r*col] = temp[r*col] + (Cap_1) * (power[r*col] + 
                    (temp[(r+1)*col] + temp[(r-1)*col] - 2.0*temp[r*col]) * Ry_1 + 
                    (temp[r*col+1] - temp[r*col]) * Rx_1 + 
                    (amb_temp - temp[r*col]) * Rz_1);
        }
        
        //if (c_end == col) 
        {
        result[r*col+col-1] = temp[r*col+col-1] +(Cap_1) * (power[r*col+col-1] + 
                    (temp[(r+1)*col+col-1] + temp[(r-1)*col+col-1] - 2.0*temp[r*col+col-1]) * Ry_1 + 
                    (temp[r*col+col-2] - temp[r*col+col-1]) * Rx_1 + 
                    (amb_temp - temp[r*col+col-1]) * Rz_1);
        }
        
    }
    for ( c = 0; c < col; ++c ) 
    {
        //if (r_end == row) 
        {
        result[(row-1)*col+c] =temp[(row-1)*col+c] +(Cap_1) * (power[(row-1)*col+c] + 
                    (temp[(row-1)*col+c+1] + temp[(row-1)*col+c-1] - 2.0*temp[(row-1)*col+c]) * Rx_1 + 
                    (temp[(row-2)*col+c] - temp[(row-1)*col+c]) * Ry_1 + 
                    (amb_temp - temp[(row-1)*col+c]) * Rz_1);
        }
        
        //if (r_start == 0) 
        {
        result[c] =temp[c]+(Cap_1) * (power[c] + 
                (temp[c+1] + temp[c-1] - 2.0*temp[c]) * Rx_1 + 
                (temp[col+c] - temp[c]) * Ry_1 + 
                (amb_temp - temp[c]) * Rz_1);
                
        }
        
    }

    for ( r = 0; r < row; ++r ) 
    {
        kernel_ifs(teste, temp, power, (size_t)0, (size_t)col, (size_t)col, (size_t)r,(size_t) row, Cap_1, Rx_1, Ry_1, Rz_1, amb_temp);
    }
    
    result[0] = temp[0]+ (Cap_1) * (power[0] +
        (temp[1] - temp[0]) * Rx_1 +
        (temp[col] - temp[0]) * Ry_1 +
        (amb_temp - temp[0]) * Rz_1);
    //printf("Corner1\n");
    
    result[col-1] = temp[col-1]+ (Cap_1) * (power[col-1] +
        (temp[col-2] - temp[col-1]) * Rx_1 +
        (temp[2*col-1] - temp[col-1]) * Ry_1 +
        (amb_temp - temp[col-1]) * Rz_1);
    //printf("Corner2\n");
    
    result[(row-1)*col+col-1] =temp[(row-1)*col+col-1] + (Cap_1) * (power[(row-1)*col+col-1] + 
        (temp[(row-1)*col+col-2] - temp[(row-1)*col+col-1]) * Rx_1 + 
        (temp[(row-2)*col+col-1] - temp[(row-1)*col+col-1]) * Ry_1 + 
        (amb_temp - temp[(row-1)*col+col-1]) * Rz_1);	
    //printf("Corner3\n");						

    result[(row-1)*col] =temp[(row-1)*col] + (Cap_1) * (power[(row-1)*col] + 
        (temp[(row-1)*col+1] - temp[(row-1)*col]) * Rx_1 + 
        (temp[(row-2)*col] - temp[(row-1)*col]) * Ry_1 + 
        (amb_temp - temp[(row-1)*col]) * Rz_1);
        //printf("Corner4\n");
*/
	for ( chunk = 0; chunk < num_chunk; ++chunk )
	{
        int r_start = BLOCK_SIZE_R*(chunk/chunks_in_col);
        int c_start = BLOCK_SIZE_C*(chunk%chunks_in_row); 
        int r_end = r_start + BLOCK_SIZE_R > row ? row : r_start + BLOCK_SIZE_R;
        int c_end = c_start + BLOCK_SIZE_C > col ? col : c_start + BLOCK_SIZE_C;
       
	   
        if ( r_start == 0 || c_start == 0 || r_end == row || c_end == col )
        {
			/*
			long long start_time_ifs = get_time();
			for ( r = r_start; r < r_start + BLOCK_SIZE_R; ++r ) 
			{
				kernel_ifs(result, temp, power, (size_t)c_start, (size_t)BLOCK_SIZE_C, (size_t)col, (size_t)r,(size_t) row, Cap_1, Rx_1, Ry_1, Rz_1, amb_temp);
			}
            
            long long end_time_ifs = get_time();
			total_time_ifs += ((float) (end_time_ifs - start_time_ifs)) / (1000*1000);
			continue;
			*/
			
			/*
			for ( r = r_start; r < r_start + BLOCK_SIZE_R; ++r ) 
			{
				if (c_start == 0) 
				{
				result[r*col] = temp[r*col] + (Cap_1) * (power[r*col] + 
							(temp[(r+1)*col] + temp[(r-1)*col] - 2.0*temp[r*col]) * Ry_1 + 
							(temp[r*col+1] - temp[r*col]) * Rx_1 + 
							(amb_temp - temp[r*col]) * Rz_1);
				}
				
				if (c_end == col) 
				{
				result[r*col+col-1] = temp[r*col+col-1] +(Cap_1) * (power[r*col+col-1] + 
							(temp[(r+1)*col+col-1] + temp[(r-1)*col+col-1] - 2.0*temp[r*col+col-1]) * Ry_1 + 
							(temp[r*col+col-2] - temp[r*col+col-1]) * Rx_1 + 
							(amb_temp - temp[r*col+col-1]) * Rz_1);
				}
				
			}
			for ( c = c_start; c < c_start + BLOCK_SIZE_C; ++c ) 
			{
				if (r_end == row) 
				{
				result[(row-1)*col+c] =temp[(row-1)*col+c] +(Cap_1) * (power[(row-1)*col+c] + 
							(temp[(row-1)*col+c+1] + temp[(row-1)*col+c-1] - 2.0*temp[(row-1)*col+c]) * Rx_1 + 
							(temp[(row-2)*col+c] - temp[(row-1)*col+c]) * Ry_1 + 
							(amb_temp - temp[(row-1)*col+c]) * Rz_1);
				}
				
				if (r_start == 0) 
				{
				result[c] =temp[c]+(Cap_1) * (power[c] + 
						(temp[c+1] + temp[c-1] - 2.0*temp[c]) * Rx_1 + 
						(temp[col+c] - temp[c]) * Ry_1 + 
						(amb_temp - temp[c]) * Rz_1);
						
				}
				
			}
			*/
			
			long long start_time_ifs = get_time();
			for ( r = r_start; r < r_start + BLOCK_SIZE_R; ++r ) 
			{
				for ( c = c_start; c < c_start + BLOCK_SIZE_C; ++c ) 
				{
					if ( (r == 0) && (c == 0) ) {
                        delta = (Cap_1) * (power[0] +
                            (temp[1] - temp[0]) * Rx_1 +
                            (temp[col] - temp[0]) * Ry_1 +
                            (amb_temp - temp[0]) * Rz_1);
                    }   
                    else if ((r == 0) && (c == col-1)) {
                        delta = (Cap_1) * (power[c] +
                            (temp[c-1] - temp[c]) * Rx_1 +
                            (temp[c+col] - temp[c]) * Ry_1 +
                        (   amb_temp - temp[c]) * Rz_1);
                    }  
                    else if ((r == row-1) && (c == col-1)) {
                        delta = (Cap_1) * (power[r*col+c] +
                            (temp[r*col+c-1] - temp[r*col+c]) * Rx_1 +
                            (temp[(r-1)*col+c] - temp[r*col+c]) * Ry_1 +
                        (   amb_temp - temp[r*col+c]) * Rz_1);
                    }   
                    else if ((r == row-1) && (c == 0)) {
                        delta = (Cap_1) * (power[r*col] +
                            (temp[r*col+1] - temp[r*col]) * Rx_1 +
                            (temp[(r-1)*col] - temp[r*col]) * Ry_1 +
                            (amb_temp - temp[r*col]) * Rz_1);
                    }   
                    else if (r == 0) {
                        delta = (Cap_1) * (power[c] +
                            (temp[c+1] + temp[c-1] - 2.0*temp[c]) * Rx_1 +
                            (temp[col+c] - temp[c]) * Ry_1 +
                            (amb_temp - temp[c]) * Rz_1);
                    }   
                    else if (c == col-1) {
                        delta = (Cap_1) * (power[r*col+c] +
                            (temp[(r+1)*col+c] + temp[(r-1)*col+c] - 2.0*temp[r*col+c]) * Ry_1 +
                            (temp[r*col+c-1] - temp[r*col+c]) * Rx_1 +
                            (amb_temp - temp[r*col+c]) * Rz_1);
                    }  
                    else if (r == row-1) {
                        delta = (Cap_1) * (power[r*col+c] +
                            (temp[r*col+c+1] + temp[r*col+c-1] - 2.0*temp[r*col+c]) * Rx_1 +
                            (temp[(r-1)*col+c] - temp[r*col+c]) * Ry_1 +
                            (amb_temp - temp[r*col+c]) * Rz_1);
                    }   
                    else if (c == 0) {
                        delta = (Cap_1) * (power[r*col] +
                            (temp[(r+1)*col] + temp[(r-1)*col] - 2.0*temp[r*col]) * Ry_1 +
                            (temp[r*col+1] - temp[r*col]) * Rx_1 +
                            (amb_temp - temp[r*col]) * Rz_1);
                    }
                    result[r*col+c] =temp[r*col+c]+ delta;
				}
			}
			long long end_time_ifs = get_time();
			total_time_ifs += ((float) (end_time_ifs - start_time_ifs)) / (1000*1000);
			continue;
			
			/*
			result[0] = temp[0]+ (Cap_1) * (power[0] +
				(temp[1] - temp[0]) * Rx_1 +
				(temp[col] - temp[0]) * Ry_1 +
				(amb_temp - temp[0]) * Rz_1);
			//printf("Corner1\n");
			
			result[col-1] = temp[col-1]+ (Cap_1) * (power[col-1] +
				(temp[col-2] - temp[col-1]) * Rx_1 +
				(temp[2*col-1] - temp[col-1]) * Ry_1 +
				(amb_temp - temp[col-1]) * Rz_1);
			//printf("Corner2\n");
			
			result[(row-1)*col+col-1] =temp[(row-1)*col+col-1] + (Cap_1) * (power[(row-1)*col+col-1] + 
				(temp[(row-1)*col+col-2] - temp[(row-1)*col+col-1]) * Rx_1 + 
				(temp[(row-2)*col+col-1] - temp[(row-1)*col+col-1]) * Ry_1 + 
				(amb_temp - temp[(row-1)*col+col-1]) * Rz_1);	
			//printf("Corner3\n");						

			result[(row-1)*col] =temp[(row-1)*col] + (Cap_1) * (power[(row-1)*col] + 
				(temp[(row-1)*col+1] - temp[(row-1)*col]) * Rx_1 + 
				(temp[(row-2)*col] - temp[(row-1)*col]) * Ry_1 + 
				(amb_temp - temp[(row-1)*col]) * Rz_1);
				//printf("Corner4\n");
				*/
			
			/*
			for ( r = r_start; r < r_start + BLOCK_SIZE_R; ++r ) 
			{
				for ( c = c_start; c < c_start + BLOCK_SIZE_C; ++c ) 
				{
					if ( ((r == 0) && (c == 0)) ||((r == 0) && (c == col-1))|| ((r == row-1) && (c == col-1)) ||((r == row-1) && (c == 0))|| (r == 0)|| (c == col-1)|| (r == row-1) || (c == 0))
					{
						if (teste[r*col+c]!=result[r*col+c])
						{
							printf("ERROU! linha:%d coluna:%d. %f, %f\n", r, c, result[r*col+c], teste[r*col+c]);
							printf("r_start:%d, r_end:%d c_start:%d, c_end:%d, col: %d, row: %d\n\n", r_start, r_end,c_start, c_end, col, row);
						}
					}
				}
			}
*/			
			
		}
		
		
        long long start_time_loop = get_time();
        for ( r = r_start; r < r_start + BLOCK_SIZE_R; ++r ) {
            kernel(result, temp, power, (size_t)c_start, (size_t)BLOCK_SIZE_C, (size_t)col, (size_t)r, Cap_1, Rx_1, Ry_1, Rz_1, amb_temp);
            //kernel(result, temp, power, (size_t)c_start, (size_t)(col-1), (size_t)col, (size_t)r, Cap_1, Rx_1, Ry_1, Rz_1, amb_temp);
        }
       long long end_time_loop = get_time();
		total_time_loop +=((float) (end_time_loop - start_time_loop)) / (1000*1000);
		
    }
	/*
	long long start_time_loop = get_time();
	for ( r = BLOCK_SIZE_R; r < row - BLOCK_SIZE_R ; ++r ) {
		//kernel(result, temp, power, (size_t)c_start, (size_t)BLOCK_SIZE_C, (size_t)col, (size_t)r, Cap_1, Rx_1, Ry_1, Rz_1, amb_temp);
		kernel(result, temp, power, (size_t)BLOCK_SIZE_C, (size_t)(col-BLOCK_SIZE_C), (size_t)col, (size_t)r, Cap_1, Rx_1, Ry_1, Rz_1, amb_temp);
	}
	long long end_time_loop = get_time();
    total_time_loop +=((float) (end_time_loop - start_time_loop)) / (1000*1000);
	*/
}

/* Transient solver driver routine: simply converts the heat 
 * transfer differential equations to difference equations 
 * and solves the difference equations by iterating
 */
void compute_tran_temp(float *result, int num_iterations, float *temp, float *power, int row, int col) 
{
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

	printf("Start computing the transient temperature\n");
	
    double start_time = get_time();

    compute_tran_temp(result,sim_time, temp, power, grid_rows, grid_cols);

    double end_time = get_time();

    printf("Ending simulation\n");
    printf("Total time: %lf\n", ((float) (end_time - start_time)) / (1000*1000));
    
    printf("Total time in ifs loop: %lf\n", total_time_ifs);
    printf("Total time in loop: %lf\n", total_time_loop);
    
    
    writeoutput((1&sim_time) ? result : temp, grid_rows, grid_cols, ofile);

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

	return 0;
}
/* vim: set ts=4 sw=4  sts=4 et si ai: */
