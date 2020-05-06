
#include "kernel.h"
#include <cstdio>
#include <stdlib.h>


#define BLOCK_SIZE 16
#define BLOCK_SIZE_C BLOCK_SIZE
#define BLOCK_SIZE_R BLOCK_SIZE

void kernel_loop(float *result, float *temp, float *power, size_t c_start, size_t size, size_t col, size_t r_start,
					  float Cap_1, float Rx_1, float Ry_1, float Rz_1, float amb_temp,size_t row)
{
	size_t r;
	#define NEON_STRIDE 4
	int unroll =1;
	
	for ( r = r_start; r < row - r_start ; ++r ) {

		size_t iter = 0, rem = 0;
	
		if(size < NEON_STRIDE*unroll)
		{
			for ( int c = c_start; c < c_start + size; ++c ) 
			{
				/* Update Temperatures */
				result[r*col+c] =temp[r*col+c]+ ( Cap_1 * (power[r*col+c] + 
					(temp[(r+1)*col+c] + temp[(r-1)*col+c] - 2.f*temp[r*col+c]) * Ry_1 + 
					(temp[r*col+c+1] + temp[r*col+c-1] - 2.f*temp[r*col+c]) * Rx_1 + 
					(amb_temp - temp[r*col+c]) * Rz_1));
			}
			
			 return;
		}
		iter = (size+c_start) / (NEON_STRIDE*unroll) * (NEON_STRIDE*unroll);

	
		
		asm volatile (
			 "lsl x1, %[c], #2 \n\t"				//c=c_start
			 "lsl x2, %[r], #2 \n\t"

			 "ld1r { v4.4s } , [%[ca]]\n\t"
			 "mul x2, x2, %[col]\n\t"				//r*col
			 "fmov v9.4s , #2\n\t"
			 "sub x3, x2, #4\n\t"					//r*col-1
			 "add x4, %[temp], x2\n\t"				//x4, *temp[r*col]
			 "add x5, %[pow], x2\n\t"				//x5, *power[r*col]
			 "add x6, %[temp], x3\n\t"				//x6, *temp[r*col-1]
			 "add x3, x2, #4\n\t"					//r*col+1
			 "ld1r { v1.4s } , [%[Ry]]\n\t"
			 "add x7, %[temp], x3\n\t"				//x7, *temp[r*col+1]
			 "add x3, x2, %[col], LSL #2\n\t"		//(r+1)*col
			 "ld1r { v0.4s } , [%[Rx]]\n\t"
			 "add x9, %[temp], x3\n\t"				//x9, *temp[(r+1)*col]
			 "sub x3, x2, %[col], LSL #2\n\t"		//(r-1)*col
			 "ld1r { v2.4s } , [%[Rz]]\n\t"
			 "ld1r { v3.4s } , [%[amb]]\n\t"
			 "add x10, %[temp], x3\n\t"				//x10,*temp[(r-1)*col]
			 "add x11, %[res], x2\n\t"				//x11,*result[r*col]
			 
			
			".loop_neon:\n\t"
			 "ldr q5, [x4, x1]\n\t"					//temp[r*col+c]
			 "ldr q8, [x6, x1]\n\t"					//v8 auxiliar, temp[r*col+c-1]
			 "fsub v6.4s, v3.4s, v5.4s\n\t"			//v6 auxiliar, (amb_temp - temp[r*col+c])
			 "ldr q10, [x7, x1]\n\t"				//v10 auxiliar, temp[r*col+c+1]
			 "fmul v7.4s, v6.4s, v2.4s\n\t"			//v7 acumulador
			 "fadd v6.4s, v10.4s, v8.4s\n\t"		//v6 auxiliar, temp[r*col+c+1]+temp[r*col+c-1]
			 "ldr q8, [x10, x1]\n\t"				//v8 auxiliar, temp[(r-1)*col+c]
			 "fmls v6.4s, v5.4s, v9.4s\n\t"			//v6 auxiliar, (temp[r*col+c+1] + temp[r*col+c-1] - 2.f*temp[r*col+c])
			 "fmla v7.4s, v6.4s, v0.4s\n\t"			//v7 acumulador 
			 "ldr q6, [x9, x1]\n\t"					//v6 auxiliar, temp[(r+1)*col+c]
			 "fadd v6.4s, v6.4s, v8.4s\n\t"			//v6 auxiliar, temp[(r+1)*col+c]+temp[(r-1)*col+c]
			 "ldr q10, [x5, x1]\n\t"				//v10 auxiliar, power[r*col+c]
			 "fmls v6.4s, v5.4s, v9.4s\n\t"			//v6 auxiliar, (temp[(r+1)*col+c]+temp[(r-1)*col+c] - 2.f*temp[r*col+c])
			 "fmla v7.4s, v6.4s, v1.4s\n\t"			//v7 acumulador
			 "fadd v8.4s, v10.4s, v7.4s\n\t"		//v8 auxiliar, acumulador(v7)+power[r+*col+c]
			 "fmla v5.4s, v8.4s, v4.4s\n\t"			//result[r*col+c]
			 "str q5, [x11, x1]\n\t"
			 "add x1, x1, #16\n\t"					//iterador+4
			 "cmp x1, %[sz]\n\t"
			 "b.ne .loop_neon\n\t"
			
			 : [res] "+r" (result)
			 : [c] "r" (c_start), [Rx] "r" (&Rx_1), [Ry] "r" (&Ry_1), [Rz] "r" (&Rz_1), [amb] "r" (&amb_temp), [ca] "r" (&Cap_1), [temp] "r" (temp),
			 [pow] "r" (power), [r] "r" (r), [col] "r" (col), [sz] "r" (iter*4)
			 : "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x9", "x10", "x11", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10"
		);
			
		
		
		rem = (size+c_start) % (NEON_STRIDE*unroll);
		
		for ( int c = iter; c < rem + iter; ++c ) 
		{
			result[r*col+c] =temp[r*col+c]+ ( Cap_1 * (power[r*col+c] + 
				(temp[(r+1)*col+c] + temp[(r-1)*col+c] - 2.f*temp[r*col+c]) * Ry_1 + 
				(temp[r*col+c+1] + temp[r*col+c-1] - 2.f*temp[r*col+c]) * Rx_1 + 
				(amb_temp - temp[r*col+c]) * Rz_1));
				
		}
	}
}

void kernel_ifs(float *result, float *temp, float *power, size_t c_start, size_t size, size_t col, size_t r, size_t row, float Cap_1, float Rx_1, 
				float Ry_1, float Rz_1, float amb_temp, float *delta, int num_chunk, int chunks_in_row, int chunks_in_col)
{
	
	for ( int chunk = 0; chunk < num_chunk; ++chunk )
	{
		int r_start = BLOCK_SIZE_R*(chunk/chunks_in_col);
		int c_start = BLOCK_SIZE_C*(chunk%chunks_in_row); 
		int r_end = r_start + BLOCK_SIZE_R > row ? row : r_start + BLOCK_SIZE_R;
		int c_end = c_start + BLOCK_SIZE_C > col ? col : c_start + BLOCK_SIZE_C;
	   
	   
		if ( r_start == 0 || c_start == 0 || r_end == row || c_end == col )
		{	
			
			for ( r = r_start; r < r_start + BLOCK_SIZE_R; ++r ) 
			{
				//NEON
				asm volatile (
					 //if (r==0)
					 "cmp %[r], #0\n\t"
					 "b.eq .neon_r_0\n\t"
					 
					 //if (r==row-1)
					 "sub x1, %[row], #1\n\t"
					 "cmp %[r], x1\n\t"
					 "b.eq .neon_r_end\n\t"
					 
					 //if (c ==0)
					 "lsl x1, %[c], #2\n\t"								//c =c_start caso c!0
					 "cmp x1, #0\n\t"
					 "b.ne .neon_normal\n\t"
					 
					 //c =0
					 "lsl x1, %[r], #2\n\t"								//r
					 "mul x1, x1, %[col]\n\t"							//r*col
					 "ldr s1, [%[amb]]\n\t"								//amb_temp
					 "ldr s5, [%[temp], x1]\n\t"						//temp[r*col]
					 "ldr s2, [%[Rx]]\n\t"								//Rx_1
					 "ldr s3, [%[Ry]]\n\t"								//Ry_1
					 "ldr s4, [%[Rz]]\n\t"								//Rz_1
					 "fsub s1, s1, s5\n\t"								//amb_temp - temp[r*col]
					 "add x2, x1, #4\n\t"								///r*col+1
					 "fmul s1, s4, s1\n\t"
					 "ldr s6, [%[temp], x2]\n\t"						//temp[r*col+1]
					 "fsub s6, s6, s5\n\t"								//temp[r*col+1] - temp[r*col]
					 "ldr s4, [%[pow], x1]\n\t"							//power[r*col]
					 "fmadd s1, s6, s2, s1\n\t"							//acumulador
					 "add x2, x1, %[col], lsl #2\n\t"					//(r+1)*col
					 "ldr s6, [%[temp], x2]\n\t"						//temp[(r+1)*col]
					 "sub x2, x1, %[col], lsl #2\n\t"					//(r-1)*col
					 "fmov s7, #2\n\t"
					 "ldr s2, [%[temp], x2]\n\t"						//temp[(r-1)*col]
					 "ldr s8, [%[ca]]\n\t"								//cap_1
					 "fadd s6, s6, s2\n\t"								//temp[(r+1)*col]+ temp[(r-1)*col]
					 "fmsub s6, s7, s5, s6\n\t"							//temp[(r+1)*col] + temp[(r-1)*col] - 2.0*temp[r*col]
					 "fmadd s1, s6, s3, s1\n\t"							//acumulador
					 "fadd s1, s4, s1\n\t"								//acumulador+power[r*col]
					 "fmul s0, s1, s8\n\t"								//delta  
					 "str s0, [%[delta]]\n\t"
					 "fadd s1, s0, s5\n\t"								//result[r*col]
					 "str s1, [%[res], x1]\n\t"
					 "mov x1, #4\n\t"									//c=1*4
					
					
					 ".neon_normal:\n\t"
					 //x1 iterador c=c_start || c=1
					 "lsl x2, %[col], #2\n\t"								//x2, col*4
					 "mul x2, x2, %[r]\n\t"							//x2, r*col*4
					 "add x2, x2, x1\n\t"					//(r*col+c)*4
					 "ld1r {v2.4s} , [%[delta]]\n\t"                             //q2, delta
					 "dup s0, v2.4s[3]\n\t"                       // s0 delta, save last delta
					 //loop 
					 ".loop_neon_normal:\n\t"
					 "ldr q1 , [%[temp], x2]\n\t"
					 "fadd v1.4s, v1.4s, v2.4s\n\t"					    //temp[r*col+c]+delta
					 "str q1, [%[res], x2]\n\t" 
					 "add x2, x2, #16\n\t"				    	        //r*col+c+4
					 "add x1, x1, #16\n\t"				    	        //iterador+4
					 "cmp x1, %[sz]\n\t"
					 "b.le .loop_neon_normal\n\t"
					 "sub x2, %[col], #1\n\t"                           // 
					 "lsl x2, x2, #2\n\t"                               //(col - 1)*4
					 "cmp x1, x2\n\t"
					 "b.ne .neon_end\n\t"
					
					
					 
					 //c=col-1
					 "lsl x1, %[col], #2\n\t"								//c=col*4 FIXME
					 "sub x1, x1, #4\n\t"								//c=(col-1)*4
					 "lsl x2, %[r], #2 \n\t"							//r*4
					 "madd x2, x2, %[col], x1\n\t"						//r*col*4+c*4
					 "ldr s1, [%[amb]]\n\t"								//amb_temp
					 "ldr s5, [%[temp], x2]\n\t"						//temp[r*col+c]
					 "ldr s2, [%[Rx]]\n\t"								//Rx_1
					 "ldr s3, [%[Ry]]\n\t"								//Ry_1
					 "ldr s4, [%[Rz]]\n\t"								//Rz_1
					 "fsub s1, s1, s5\n\t"								//amb_temp - temp[r*col+c]
					 "sub x3, x2, #4\n\t"								///r*col+c-1
					 "fmul s1, s4, s1\n\t"
					 "ldr s6, [%[temp], x3]\n\t"						//temp[r*col+c-1]
					 "fsub s6, s6, s5\n\t"								//temp[r*col+c-1] - temp[r*col+c]
					 "ldr s4, [%[pow], x2]\n\t"							//power[r*col]
					 "fmadd s1, s6, s2, s1\n\t"							//acumulador
					 "add x3, x2, %[col], lsl #2\n\t"					//(r+1)*col+c
					 "ldr s6, [%[temp], x3]\n\t"						//temp[(r+1)*col+c]
					 "sub x3, x2, %[col], lsl #2\n\t"					//(r-1)*col+c
					 "fmov s7, #2\n\t"
					 "ldr s2, [%[temp], x3]\n\t"						//temp[(r-1)*col]
					 "ldr s8, [%[ca]]\n\t"								//cap_1
					 "fadd s6, s6, s2\n\t"								//temp[(r+1)*col+c] + temp[(r-1)*col+c]
					 "fmsub s6, s7, s5, s6\n\t"							//temp[(r+1)*col+c] + temp[(r-1)*col+c] - 2.0*temp[r*col+c]
					 "fmadd s1, s6, s3, s1\n\t"							//acumulador
					 "fadd s1, s4, s1\n\t"								//acumulador+power[r*col+c]
					 "fmul s0, s1, s8\n\t"								//delta  
					 "fadd s1, s0, s5\n\t"								//result[r*col+c]
					 "str s1, [%[res], x2]\n\t"
					 "b .neon_end\n\t"									//COMFIRMAR NOME
					
					 //r=0
					 ".neon_r_0:\n\t"
					 "lsl x1, %[c], #2\n\t"								//iterador c=c_start
					 
					 "ld1r { v10.4s }, [%[Rx]]\n\t"
					 "ld1r { v1.4s }, [%[Ry]]\n\t"
					 "ld1r { v2.4s }, [%[Rz]]\n\t"
					 "ld1r { v3.4s }, [%[amb]]\n\t"
					 "ld1r { v4.4s }, [%[ca]]\n\t"
					 "fmov v9.4s , #2\n\t"
					 "add x4, %[temp], %[col], lsl #2\n\t"              //x4, *temp[col]
					 "sub x5, %[temp], #4\n\t"                          //x5, *temp[-1]
					 "add x6, %[temp], #4\n\t"                          //x6, *temp[1]
						
					 //loop
					 ".loop_neon_r_0:\n\t"
					 "ldr q5 , [%[temp], x1]\n\t"	                    //q5, temp[c]
					 "fsub v6.4s, v3.4s, v5.4s\n\t"					    //v6, (amb_temp - temp[c])
					 "ldr q7 , [ x4, x1] \n\t"	                        //q7, temp[col+c]
					 "fmul v6.4s,  v6.4s, v2.4s\n\t"					//v6, (amb_temp - temp[c])*Rz_1
					 "fsub v7.4s , v7.4s, v5.4s\n\t"					//v7, temp[col+c]-temp[c]
					 "fmla v6.4s , v7.4s, v1.4s\n\t"					//v6, acumulador
					 "ldr q7 , [ x6, x1]\n\t"	                        //q7, temp[c+1]
					 "ldr q8 , [ x5,x1]\n\t"	                        //q8, temp[c-1]
					 "fadd v7.4s, v7.4s, v8.4s\n\t"					    //v7, temp[c+1]+temp[c-1]
					 "fmls v7.4s, v9.4s, v5.4s\n\t"					    //v7,(temp[c+1]+temp[c-1] - 2.0*temp[c])
					 "fmul v11.4s,  v7.4s, v10.4s\n\t"					//FIXME
					 "fadd v6.4s, v6.4s, v11.4s\n\t"					//FIXME
					 "ldr q8 , [%[pow], x1]\n\t"	                    //q8, pow[c]
					 "fadd v8.4s, v8.4s, v6.4s\n\t"					    //v8, acumulador(v6)+power[c]
					 "fmul v8.4s, v8.4s, v4.4s\n\t"					    //delta
					 "dup s0, v8.4s[3]\n\t"                       // s0 delta, save last delta
					 "fadd v5.4s, v5.4s, v8.4s\n\t"					    //v5 acumulador
					 "str q5, [%[res], x1]\n\t"
					 "add x1, x1, #16\n\t"				    	        //iterador+4
					 "cmp x1, %[sz]\n\t"
					 "b.ne .loop_neon_r_0\n\t"
					 
					//vê se é o CORNER
					
					 "sub x2, x2, #4\n\t"							//x2=(col-1)*4
					 "cmp x1, x2\n\t"
					 "b.eq .neon_conerRU\n\t"
					 
					 "b .neon_end\n\t"
					 
					 // r=0 && c=col-1
					 ".neon_conerRU:\n\t"
					 //"lsl x1, x1, #2\n\t"								//col-1
					 "ldr s1, [%[amb]]\n\t"								//amb_temp
					 "ldr s5, [%[temp], x1]\n\t"						//temp[col-1]
					 "ldr s2, [%[Rx]]\n\t"								//Rx_1
					 "ldr s3, [%[Ry]]\n\t"								//Ry_1
					 "ldr s4, [%[Rz]]\n\t"								//Rz_1
					 "fsub s1, s1, s5\n\t"								//(amb_temp - temp[col-1])
					 "add x2, x1, %[col], lsl #2\n\t"					//col-1+col
					 "fmul s1, s4, s1\n\t"
					 "ldr s6, [%[temp], x2]\n\t"						//temp[col-1+col]
					 "fsub s6, s6, s5\n\t"								//temp[c+col] - temp[c]
					 "ldr s4, [%[ca]]\n\t"								//cap_1
					 "fmadd s1, s6, s3, s1\n\t"							//acumulador
					 "sub x2, x1, #4\n\t"								//col-1-1
					 "ldr s6, [%[temp], x2]\n\t"						//temp[col-1-1]
					 "ldr s3, [%[pow], x1]\n\t"							//power[col-1]
					 "fsub s6, s6, s5\n\t"								//temp[col-1-1]- temp[col-1]
					 "fmadd s1, s6, s2, s1\n\t"							//acumulador
					 "fadd s1, s3, s1\n\t"								//acumulador+power[col-1]
					 "fmul s0, s1, s4\n\t"								//delta
					 "fadd s1, s0, s5\n\t"								//result[col-1]
					 "str s1, [%[res], x1]\n\t"
					 "b .neon_end\n\t"									//COMFIRMAR NOME
					 
					


					 // r = row-1
					 ".neon_r_end:\n\t"	  
					 
					 "lsl x1, %[c], #2\n\t"								//iterador c=c_start
					 "lsl x2, %[r], #2\n\t"								//x2, r*4
					 "madd x2, x2, %[col], x1\n\t"					    //x2, (r*col+c)*4
					 "lsl x3, %[col], #2\n\t"							//x2 , col*4 
					 
					 "ld1r { v10.4s }, [%[Rx]]\n\t"
					 "ld1r { v1.4s }, [%[Ry]]\n\t"
					 "ld1r { v2.4s }, [%[Rz]]\n\t"
					 "ld1r { v3.4s }, [%[amb]]\n\t"
					 "ld1r { v4.4s }, [%[ca]]\n\t"
					 "fmov v9.4s , #2\n\t"
					 "sub x4, %[temp], x3\n\t"                        //x4, *temp[-col]
					 "sub x5, %[temp], #4\n\t"                        //x5, *temp[-1]
					 "add x6, %[temp], #4\n\t"                        //x6, *temp[1]
					 
					 //loop
					 ".loop_neon_r_end:\n\t"
					 "ldr q5 , [%[temp], x2]\n\t"	                    //q5, temp[r*col+c]
					 "fsub v6.4s, v3.4s, v5.4s\n\t"					    //v6, (amb_temp - temp[r*col + c])
					 "ldr q7 , [x4, x2]\n\t"	                        //q7, temp[(r-1)*col+c]
					 "fmul v6.4s,  v6.4s, v2.4s\n\t"					//v6, (amb_temp - temp[r*col+c])*Rz_1
					 "fsub v7.4s , v7.4s, v5.4s\n\t"					//v7, temp[(r-1)*col+c]-temp[r*col+c]
					 "fmla v6.4s , v7.4s, v1.4s\n\t"					//v6, acumulador
					 "ldr q7 , [ x6, x2]\n\t"	                        //q7, temp[r*col+c+1]
					 "ldr q8 , [ x5, x2]\n\t"	                        //q8, temp[r*col+c-1]
					 "fadd v7.4s, v7.4s, v8.4s\n\t"					    //v7, temp[r*col+c+1]+temp[r*col+c-1]
					 "fmls v7.4s, v9.4s, v5.4s\n\t"					    //v7,(temp[r*col+c+1]+temp[r*col+c-1] - 2.0*temp[r*col+c])
					 "fmla v6.4s, v7.4s, v10.4s\n\t"					//v6 acumulador
					 "ldr q8 , [%[pow], x2]\n\t"	                    //q8, temp[r*col+c]
					 "fadd v8.4s, v8.4s, v6.4s\n\t"					    //v8, acumulador(v6)+power[r*col+c]
					 "fmul v8.4s, v8.4s, v4.4s\n\t"					    //delta
					 "dup s0, v8.4s[3]\n\t"                       		// s0 delta, save last delta
					 "fadd v5.4s, v5.4s, v8.4s\n\t"					    //v5 acumulador
					 "str q5, [%[res], x2]\n\t"
					 "add x2, x2, #16\n\t"				    	        //r*col+c+4
					 "add x1, x1, #16\n\t"				    	        //iterador+4
					 "cmp x1, %[sz]\n\t"
					 "b.ne .loop_neon_r_end\n\t"
					
					 ".neon_end:\n\t"
					 "str s0, [%[delta]]\n\t"
					 
					 : [res] "+r" (result), [delta] "+r" (delta)
					 : [c] "r" (c_start), [Rx] "r" (&Rx_1), [Ry] "r" (&Ry_1), [Rz] "r" (&Rz_1), [amb] "r" (&amb_temp), [ca] "r" (&Cap_1), [temp] "r" (temp),
					 [pow] "r" (power), [r] "r" (r), [col] "r" (col), [row] "r" (row), [sz] "r" ((c_start+size)*4)
					 : "x1", "x2", "x3", "x4", "x5", "x6", "memory", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"
				);
			}
		}
	}
    	
}	