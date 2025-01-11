#ifndef UTILS_H
#define UTILS_H

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>

char* read_file(const char* filename, size_t* length);
void log_results(int Ilen, int Jlen, int global_size, int local_size, int success_cnt, double average_time, double ZTree_time, double ZinvTree_time, double ZinvTree_io_time, double traversej_time);
void log_t2_chunk_results(int Ilen, int Jlen, int t2_block_num, int pre_num, int success_cnt, double dec_time, double ZTree_leaf_time, double ZTree_noneleaf_time, double ZinvTree_time, double ZinvTree_io_time, double traversej_time);
void convert_char_to_int(unsigned char big_endian[32], unsigned int little_endian[8]);
void convert_int_to_char(unsigned int little_endian[8], unsigned char big_endian[32]);

#endif // UTILS_H