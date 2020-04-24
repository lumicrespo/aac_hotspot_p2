#ifndef KERNEL_AAC_HEADER__
#define KERNEL_AAC_HEADER__

#include <stddef.h>


void kernel(float *result, float *temp, float *power, size_t c_start, size_t size, size_t col, size_t r,
	   	  float Cap_1, float Rx_1, float Ry_1, float Rz_1, float amb_temp);
	   	  
void kernel_ifs(float *result, float *temp, float *power, size_t c_start, size_t size, size_t col, size_t r, size_t row,
	   	  float Cap_1, float Rx_1, float Ry_1, float Rz_1, float amb_temp, float *delta);
/*
void kernel_original(float *result, float *temp, float *power, size_t c_start, size_t size, size_t col, size_t r,
					  float Cap_1, float Rx_1, float Ry_1, float Rz_1, float amb_temp);

void kernel_ifs_original(float *result, float *temp, float *power, size_t c_start, size_t size, size_t col, size_t r, size_t row,
					  float Cap_1, float Rx_1, float Ry_1, float Rz_1, float amb_temp, float *delta);

void kernel_ifs_optimized(float *result, float *temp, float *power, size_t c_start, size_t size, size_t col, size_t r, size_t row,
					  float Cap_1, float Rx_1, float Ry_1, float Rz_1, float amb_temp, float *delta);

void kernel_optimized(float *result, float *temp, float *power, size_t c_start, size_t size, size_t col, size_t r,
					  float Cap_1, float Rx_1, float Ry_1, float Rz_1, float amb_temp);
*/
#endif //KERNEL_AAC_HEADER__
