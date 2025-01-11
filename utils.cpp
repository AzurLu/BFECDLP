#include "utils.h"

char* read_file(const char* filename, size_t* length) {
    std::FILE* file = std::fopen(filename, "r");
    if (!file) {
        std::fprintf(stderr, "Failed to open file: %s\n", filename);
        std::exit(EXIT_FAILURE);
    }
    std::fseek(file, 0, SEEK_END);
    size_t size = std::ftell(file);
    std::fseek(file, 0, SEEK_SET);
    char* source = (char*)std::malloc(size + 1);
    std::fread(source, 1, size, file);
    source[size] = '\0';
    std::fclose(file);
    *length = size;
    return source;
}

void log_results(int Ilen, int Jlen, int global_size, int local_size, 
                 int success_cnt, double average_time, double ZTree_time, 
                 double ZinvTree_time, double ZinvTree_io_time, double traversej_time) {
    std::ofstream file("results_thread.csv", std::ios::app);
    if (!file) {
        std::cerr << "Failed to open file" << std::endl;
        return;
    }

    file.seekp(0, std::ios::end);
    if (file.tellp() == 0) {
        file << "Ilen,Jlen,global_size,local_size,success_cnt,average_time,"
             << "ZTree,ZinvTree,ZinvTree io,Traverse j\n";
    }

    file << Ilen << ',' << Jlen << ',' << global_size << ',' << local_size << ',' 
         << success_cnt << ',' << average_time << ',' << ZTree_time << ',' 
         << ZinvTree_time << ',' << ZinvTree_io_time << ',' << traversej_time << '\n';
}

void log_t2_chunk_results(int Ilen, int Jlen, int t2_block_num, int pre_num, int success_cnt, double dec_time, double ZTree_leaf_time, double ZTree_noneleaf_time, double ZinvTree_time, double ZinvTree_io_time, double traversej_time){
    std::ofstream file("results_t2_block_newtest.csv", std::ios::app);
    if (!file) {
        std::cerr << "Failed to open file" << std::endl;
        return;
    }

    file.seekp(0, std::ios::end);
    if (file.tellp() == 0) {
        file << "Ilen,Jlen,t2_block_num,pre_num,success_cnt,dec_time,"
             << "ZTree_leaf,ZTree_noneleaf,ZinvTree,ZinvTree io,Traverse j\n";
    }

    file << Ilen << ',' << Jlen << ',' << t2_block_num << ',' << pre_num << ',' 
         << success_cnt << ',' << dec_time << ',' << ZTree_leaf_time << ',' 
         << ZTree_noneleaf_time << ',' << ZinvTree_time << ',' << ZinvTree_io_time << ',' << traversej_time << '\n';
}

void convert_char_to_int(unsigned char big_endian[32], unsigned int little_endian[8]) {
    for (int i = 0; i < 8; i++) {
        little_endian[i] = (big_endian[(7 - i) * 4] << 24) |
            (big_endian[(7 - i) * 4 + 1] << 16) |
            (big_endian[(7 - i) * 4 + 2] << 8) |
            (big_endian[(7 - i) * 4 + 3]);
    }
}

void convert_int_to_char(unsigned int little_endian[8], unsigned char big_endian[32]) {
    for (int i = 0; i < 8; i++) {
        big_endian[(7 - i) * 4] = (little_endian[i] >> 24) & 0xFF;
        big_endian[(7 - i) * 4 + 1] = (little_endian[i] >> 16) & 0xFF;
        big_endian[(7 - i) * 4 + 2] = (little_endian[i] >> 8) & 0xFF;
        big_endian[(7 - i) * 4 + 3] = little_endian[i] & 0xFF;
    }
}