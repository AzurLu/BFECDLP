#define CL_TARGET_OPENCL_VERSION 300

#include "./secp256k1/src/secp256k1.c"
#include "./secp256k1/include/secp256k1.h"
#include "./secp256k1/src/scalar_impl.h"
#include "./secp256k1/src/field_impl.h"
#include "./secp256k1/src/group_impl.h"
#include "./secp256k1/src/ecmult_gen_impl.h"
#include "./secp256k1/src/precomputed_ecmult.c"
#include "./secp256k1/src/precomputed_ecmult_gen.c"
#include "threadpool.h"
#include "utils.h"
#include "random.h"
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <vector>
#include <string>
#include <chrono>
#include <CL/cl.h>

static int pre_num;
static std::atomic<bool> stop_thread_flag{false};
std::atomic<bool> processing_done(false);
std::mutex queue_mutex;
std::condition_variable cv;

void create_ZTree_leafnode(int start, int end, std::vector<secp256k1_fe> &ZTree, secp256k1_fe *t2, const secp256k1_ge &mG_affine){
    for (int i = start; i < end; i++) {
        ZTree[i] = mG_affine.x;
        secp256k1_fe t2x_fe_neg;
        secp256k1_fe_negate(&t2x_fe_neg, &t2[(i + 1) * 2], 1);
        secp256k1_fe_add(&ZTree[i], &t2x_fe_neg); // x_tmp = mG - t2x
        secp256k1_fe_normalize(&ZTree[i]);
    }
}

void create_ZTree_nonleafnode(int start, int end, int pre_start, int cur_start, std::vector<secp256k1_fe> &ZTree){
    int pre_index = pre_start + (start - cur_start) * 2;
    for(int i = start; i < end; i++){
        secp256k1_fe_mul(&ZTree[i], &ZTree[pre_index], &ZTree[pre_index + 1]);
        secp256k1_fe_normalize(&ZTree[i]);
        pre_index += 2;
    }
}

void create_ZinvTree(int start, int end, int pre_start, int cur_start, std::vector<secp256k1_fe> &ZTree, secp256k1_fe *ZinvTree){
    int pre_index = pre_start + (start - cur_start) * 2;
    for(int i = start; i < end; i++){
        secp256k1_fe_mul(&ZinvTree[pre_index], &ZinvTree[i], &ZTree[pre_index ^ 1]);
        secp256k1_fe_normalize(&ZinvTree[pre_index]);
        secp256k1_fe_mul(&ZinvTree[pre_index ^ 1], &ZinvTree[i], &ZTree[pre_index]);
        secp256k1_fe_normalize(&ZinvTree[pre_index ^ 1]);
        pre_index += 2;
    }
}

void read_t2_block(std::ifstream &read_file, int block_idx, int block_size, std::deque<cl_mem> &block_queue, cl_context &context, cl_command_queue &queue){
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        cv.wait(lock, [&]() { return stop_thread_flag.load() || block_queue.size() <= pre_num; });
    }
    if(stop_thread_flag.load()){
        cv.notify_all();
        return;
    }
    cl_mem buf_t2_block = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, static_cast<size_t>((block_size + 1) * 2 * sizeof(secp256k1_fe)), NULL, NULL);
    secp256k1_fe* t2_block = (secp256k1_fe *)clEnqueueMapBuffer(queue, buf_t2_block, CL_TRUE, CL_MAP_WRITE, 0, static_cast<size_t>((block_size + 1) * 2 * sizeof(secp256k1_fe)), 0, NULL, NULL, NULL);
    secp256k1_fe_set_int(&t2_block[0], block_idx);
    
    if(!read_file.read(reinterpret_cast<char*>(t2_block + 2), block_size * 2 * sizeof(secp256k1_fe))){
        std::cerr << "Failed to read t2_block" << std::endl;
    }

    clEnqueueUnmapMemObject(queue, buf_t2_block, t2_block, 0, NULL, NULL);
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        block_queue.push_back(buf_t2_block);
        // printf("Queue pushed. Current size: %d. Current id: %d\n", block_queue.size(), block_idx);
    }
    cv.notify_all();
}


int main(int argc, char *argv[]) {
    int Ilen = atoi(argv[1]);
    int Jlen = atoi(argv[2]);
    cl_int Imax = 1 << Ilen;
    cl_int Jmax = 1 << Jlen;

    secp256k1_context* ctx;
    ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN);

    unsigned char seckey[32];
    secp256k1_scalar seckey_scalar;
    while (1) {
      fill_random(seckey, sizeof(seckey));
      if (secp256k1_ec_seckey_verify(ctx, seckey)) break;
    }
    secp256k1_scalar_set_b32_seckey(&seckey_scalar, seckey);
    
    secp256k1_pubkey pubkey;
    int created = secp256k1_ec_pubkey_create(ctx, &pubkey, seckey);
    

    // OpenCL initialize
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_int err;

    size_t global_size;
    size_t local_size;

    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) {
        printf("clGetPlatformIDs");
    }

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        printf("clGetDeviceIDs");
    }

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("clCreateContext");
    }

    queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
    if (err != CL_SUCCESS) {
        printf("clCreateCommandQueueWithProperties");
    }

    // Read kernel source file
    size_t source_size;
    char* source = read_file("bfecdlp_chunkio_main.cl", &source_size);

    program = clCreateProgramWithSource(context, 1, (const char**)&source, &source_size, &err);
    if (err != CL_SUCCESS) {
        printf("clCreateProgramWithSource");
    }
    free(source);

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    size_t log_size;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    char* log = (char*)malloc(log_size);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
    // printf("Build Log:\n%s\n", log);
    free(log);

    // t1 start
    cl_uint cuckoolen = Imax * 1.3;
    std::vector<cl_uchar> XS((size_t)Imax * 32);
    std::vector<cl_uint> table_k(cuckoolen, 0);
    std::vector<cl_int> table_v(cuckoolen, 0);
    std::vector<cl_uchar> hash_index(cuckoolen, 0);

    char t1_filename[10];
    std::snprintf(t1_filename, sizeof(t1_filename), "t1_%d", Ilen);
    std::ifstream file(t1_filename, std::ios::binary);
    if (file){
        if (!file.read(reinterpret_cast<char*>(table_k.data()), sizeof(cl_uint) * cuckoolen)) {
            std::cerr << "Failed to read table_k" << std::endl;
            return 0;
        }

        if (!file.read(reinterpret_cast<char*>(table_v.data()), sizeof(cl_int) * cuckoolen)) {
            std::cerr << "Failed to read table_v" << std::endl;
            return 0;
        }

        // printf("t1 deserialize done\n");
    }
    else{
        secp256k1_gej ig; secp256k1_gej_set_ge(&ig, &secp256k1_ge_const_g);
        secp256k1_ge cur_ig_affine;
        cl_uchar xbuf[32], ybuf[32];
        for (int i = 0; i < Imax; i++) {
            secp256k1_ge_set_gej(&cur_ig_affine, &ig);
            secp256k1_fe_normalize(&(cur_ig_affine.x));
            secp256k1_fe_get_b32(xbuf, &(cur_ig_affine.x));
            for (int j = 0; j < 32; j++) {
                XS[i * 32 + j] = xbuf[j];
            }
            secp256k1_gej_add_ge(&ig, &ig, &secp256k1_ge_const_g);
        }
        printf("XS calculation done\n");

        int maxiter = 500;
        for (int i = 0; i < Imax; i++) {
            int v = i + 1;
            cl_uchar old_hash_id = 1;

            int j = 0;
            for (; j < maxiter; j++) {
                long long index = (long long)(v - 1) * 32 + (old_hash_id - 1) * 8;
                cl_uint h = ((cl_uint)XS[index] << 24 |
                            (cl_uint)XS[index + 1] << 16 |
                            (cl_uint)XS[index + 2] << 8 |
                            (cl_uint)XS[index + 3]) % cuckoolen;
                cl_uint x = (cl_uint)XS[index + 4] << 24 |
                            (cl_uint)XS[index + 5] << 16 |
                            (cl_uint)XS[index + 6] << 8 |
                            (cl_uint)XS[index + 7];
                cl_uint x_key = x;
                if (hash_index[h] == 0) {
                    hash_index[h] = old_hash_id;
                    table_v[h] = v;
                    table_k[h] = x_key;
                    break;
                }
                else {
                    cl_int temp1 = v;
                    v = table_v[h];
                    table_v[h] = temp1;

                    cl_uchar temp2 = old_hash_id;
                    old_hash_id = hash_index[h];
                    hash_index[h] = temp2;

                    cl_uint temp3 = x_key;
                    x_key = table_k[h];
                    table_k[h] = temp3;

                    old_hash_id = old_hash_id % 3 + 1;
                }
            }
            if (j == maxiter - 1) {
                printf("insert failed, %d\n", i);
            }
        }
        
        std::ofstream out_file(t1_filename, std::ios::binary);
        if (!out_file) {
            std::cerr << "Failed to open file for writing: " << t1_filename << std::endl;
            return 0;
        }
        if (!out_file.write(reinterpret_cast<const char*>(table_k.data()), sizeof(cl_uint) * cuckoolen)) {
            std::cerr << "Failed to write table_k" << std::endl;
            return 0;
        }
        if (!out_file.write(reinterpret_cast<const char*>(table_v.data()), sizeof(cl_int) * cuckoolen)) {
            std::cerr << "Failed to write table_v" << std::endl;
            return 0;
        }

        // printf("t1 serialize done\n");
    }
    // Create and write in buf_table_k, buf_table_v
    cl_mem buf_table_k = clCreateBuffer(context, CL_MEM_READ_WRITE, (size_t)cuckoolen * sizeof(cl_uint), NULL, &err);
    if (err != CL_SUCCESS) {
        printf("clCreateBuffer");
    }
    cl_mem buf_table_v = clCreateBuffer(context, CL_MEM_READ_WRITE, (size_t)cuckoolen * sizeof(cl_int), NULL, &err);
    if (err != CL_SUCCESS) {
        printf("clCreateBuffer");
    }
    err = clEnqueueWriteBuffer(queue, buf_table_k, CL_TRUE, 0, (size_t)cuckoolen * sizeof(cl_uint), table_k.data(), 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("clEnqueueWriteBuffer");
    }
    err = clEnqueueWriteBuffer(queue, buf_table_v, CL_TRUE, 0, (size_t)cuckoolen * sizeof(cl_int), table_v.data(), 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("clEnqueueWriteBuffer");
    }
    
    XS.clear();
    XS.shrink_to_fit();
    table_k.clear();
    table_k.shrink_to_fit();
    table_v.clear();
    table_v.shrink_to_fit();
    hash_index.clear();
    hash_index.shrink_to_fit();
    // printf("t1 done\n");
    
    
    // t2 start
    int t2_block_num = atoi(argv[3]);
    pre_num = atoi(argv[4]);
    int block_size = Jmax / t2_block_num;
    int temp = block_size, width = 0;
    while(temp > 1){
        temp >>= 1;
        ++width;
    }
    unsigned int threads_num = 0;
    std::deque<cl_mem> block_queue;
    ThreadPool inner_pool(40);
    std::vector<std::future<void>> inner_futures;
    ThreadPool pre_pool(1);
    std::vector<std::future<void>> futures;
    
    char t2_filename[10];
    std::snprintf(t2_filename, sizeof(t2_filename), "t2_%d_%d", Ilen, Jlen);
    std::ifstream t2_file(t2_filename, std::ios::binary);
    if(!t2_file){
        std::unique_ptr<secp256k1_fe[]> t2(new secp256k1_fe[(Jmax + 1) * 2]);
        secp256k1_scalar imax_scalar; secp256k1_scalar_set_int(&imax_scalar, 2 * Imax); // potential segmentation fault
        secp256k1_gej twoimaxg_jac;
        secp256k1_ge twoimaxg_affine;
        secp256k1_gej j2imaxg;
        secp256k1_ge cur_j2imaxg_affine;
        secp256k1_ecmult_gen(&ctx->ecmult_gen_ctx, &twoimaxg_jac, &imax_scalar);
        secp256k1_ge_set_gej(&twoimaxg_affine, &twoimaxg_jac);
        secp256k1_gej_set_ge(&j2imaxg, &twoimaxg_affine);
        for (int j = 1; j <= Jmax; j++) {
            // printf("%d ", j);
            secp256k1_ge_set_gej(&cur_j2imaxg_affine, &j2imaxg);
            secp256k1_fe_normalize(&(cur_j2imaxg_affine.x));
            secp256k1_fe_normalize(&(cur_j2imaxg_affine.y));
            t2[j * 2] = cur_j2imaxg_affine.x;
            t2[j * 2 + 1] = cur_j2imaxg_affine.y;
            secp256k1_gej_add_ge(&j2imaxg, &j2imaxg, &twoimaxg_affine);
        }

        std::ofstream out_file(t2_filename, std::ios::binary);
        if(!out_file){
            std::cerr << "Failed to open file for writing: " << t2_filename << std::endl;
            return 0;
        }
        if(!out_file.write(reinterpret_cast<const char *>(t2.get()), sizeof(secp256k1_fe) * (Jmax + 1) * 2)){
            std::cerr << "Failed to write t2" << std::endl;
            return 0;
        }

        // printf("t2 serialize done\n");
    }
    t2_file.close();

    // random test start
    int mlen = Ilen + Jlen + 1;
    cl_uchar m[32];
    cl_uint affmGx[8], affmGy[8];
    cl_uint Treelen = block_size * 2;
    std::vector<secp256k1_fe> ZTree(Treelen);
    cl_int stop_flag = -1;
    cl_uint m_dec[8] = { 0 };
    int ite = 10;

    long long sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0, sum5 = 0, sum_dec = 0;

    //started from Jlen = 11
    int ZTree_leafnode_threshold = 1 << 11;
    unsigned int ZTree_leafnode_threadsnum[20] = {6, 6, 8, 16, 16, 23, 26, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40};
    //started from cur_width = 10
    int ZTree_nonleafnode_threshold = 1 << 10;
    unsigned int ZTree_nonleafnode_threadsnum[20] = {6, 6, 12, 18, 21, 21, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40};
    //started from cur_width = 9
    int ZinvTree_threshold = 1 << 9;
    unsigned int ZinvTree_threadsnum[21] = {8, 8, 12, 16, 26, 28, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40};
    

    cl_mem buf_affmGx = clCreateBuffer(context, CL_MEM_READ_WRITE, 8 * sizeof(cl_uint), NULL, &err);
    if (err != CL_SUCCESS) {
        printf("clCreateBuffer buf_affmGx");
    }
    cl_mem buf_affmGy = clCreateBuffer(context, CL_MEM_READ_WRITE, 8 * sizeof(cl_uint), NULL, &err);
    if (err != CL_SUCCESS) {
        printf("clCreateBuffer buf_affmGy");
    }
    cl_mem buf_ZinvTree = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, (size_t)Treelen * sizeof(secp256k1_fe), NULL, &err);
    if (err != CL_SUCCESS) {
        printf("clCreateBuffer");
    }
    cl_mem buf_stop_flag = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(stop_flag), NULL, &err);
    if (err != CL_SUCCESS) {
        printf("clCreateBuffer");
    }
    cl_mem buf_m_dec = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(m_dec), NULL, &err);
    if (err != CL_SUCCESS) {
        printf("clCreateBuffer");
    }

    cl_kernel traversej_kernel;
    traversej_kernel = clCreateKernel(program, "Traversej", &err);
    if (err != CL_SUCCESS) {
        printf("clCreateKernel");
    }
    err = clSetKernelArg(traversej_kernel, 0, sizeof(int), &block_size);
    if (err != CL_SUCCESS) {
        printf("clSetKernelArg6-1\n");
    }
    err = clSetKernelArg(traversej_kernel, 1, sizeof(cl_int), &Ilen);
    if (err != CL_SUCCESS) {
        printf("clSetKernelArg6-2\n");
    }
    err = clSetKernelArg(traversej_kernel, 2, sizeof(cl_mem), &buf_ZinvTree);
    if (err != CL_SUCCESS) {
        printf("clSetKernelArg6-3\n");
    }
    err = clSetKernelArg(traversej_kernel, 3, sizeof(cl_mem), &buf_affmGx);
    if (err != CL_SUCCESS) {
        printf("clSetKernelArg6-4\n");
    }
    err = clSetKernelArg(traversej_kernel, 4, sizeof(cl_mem), &buf_affmGy);
    if (err != CL_SUCCESS) {
        printf("clSetKernelArg6-5\n");
    }
    err = clSetKernelArg(traversej_kernel, 6, sizeof(cl_mem), &buf_m_dec);
    if (err != CL_SUCCESS) {
        printf("clSetKernelArg6-7\n");
    }
    err = clSetKernelArg(traversej_kernel, 7, sizeof(cl_mem), &buf_stop_flag);
    if (err != CL_SUCCESS) {
        printf("clSetKernelArg6-8\n");
    }
    err = clSetKernelArg(traversej_kernel, 8, sizeof(cl_mem), &buf_table_k);
    if (err != CL_SUCCESS) {
        printf("clSetKernelArg6-9\n");
    }
    err = clSetKernelArg(traversej_kernel, 9, sizeof(cl_mem), &buf_table_v);
    if (err != CL_SUCCESS) {
        printf("clSetKernelArg6-10\n");
    }

    int success_cnt = 0;
    for (int it = 0; it < ite; it++) {
        for (int j = 0; j < 8; j++) {
            m[31 - j] = rand() % 256;
        }
        int full_len = mlen / 8;
        int remain = mlen % 8;
        memset(m, 0, 32 - full_len - (remain > 0 ? 1 : 0));
        if (remain > 0) {
            unsigned char mask = (1 << remain) - 1;
            m[32 - full_len - 1] &= mask;
        }
        long long m_val = 0;
        for (int i = 24; i < 32; i++) {
            m_val = (m_val << 8) | m[i];
        }
        if (m_val % (2 * Imax) == 0) {
            it--;
            continue;
        }
        cl_uint m_int[8];
        convert_char_to_int(m, m_int);

        for(int i = 0; i < 8; i++) m_dec[i] = 0;
        err = clEnqueueWriteBuffer(queue, buf_m_dec, CL_TRUE, 0, sizeof(m_dec), &m_dec, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            printf("clEnqueueWriteBuffer");
        }

        stop_thread_flag.store(false);
        std::ifstream read_file(t2_filename, std::ios::binary);
        read_file.seekg(2 * sizeof(secp256k1_fe), std::ios::beg);
        
        // encrypt
        secp256k1_scalar m_scalar;
        secp256k1_scalar_set_b32(&m_scalar, m, 0);
        secp256k1_gej c1, c2;
        unsigned char nonce[32]; 
        for (int i=0; i<32; i++) nonce[i] = rand() % 256;
        secp256k1_scalar nonce_scalar;
        secp256k1_scalar_set_b32_seckey(&nonce_scalar, nonce);
        secp256k1_ecmult_gen(&ctx->ecmult_gen_ctx, &c1, &nonce_scalar); // c1 = r*G
        secp256k1_ge pk;
        secp256k1_pubkey_load(ctx, &pk, &pubkey);
        secp256k1_gej pkj;
        secp256k1_gej_set_ge(&pkj, &pk);
        secp256k1_ecmult(&c2, &pkj, &nonce_scalar, &m_scalar); // c2 = m*G+r*pk

        //decrypt
        secp256k1_gej prod;
        secp256k1_scalar zero = SECP256K1_SCALAR_CONST(0, 0, 0, 0, 0, 0, 0, 0);
        secp256k1_ecmult(&prod, &c1, &seckey_scalar, &zero); // sk * c1
        secp256k1_gej_neg(&prod, &prod); // -sk * c1
        secp256k1_gej mG;
        secp256k1_gej_add_var(&mG, &c2, &prod, NULL);
        secp256k1_ge mG_affine; secp256k1_ge_set_gej(&mG_affine, &mG);

        cl_uchar xbuf[32], ybuf[32];
        secp256k1_fe_normalize(&(mG_affine.x));
        secp256k1_fe_normalize(&(mG_affine.y));
        secp256k1_fe_get_b32(xbuf, &(mG_affine.x));
        secp256k1_fe_get_b32(ybuf, &(mG_affine.y));
        convert_char_to_int(xbuf, affmGx);
        convert_char_to_int(ybuf, affmGy);
        err = clEnqueueWriteBuffer(queue, buf_affmGx, CL_TRUE, 0, 8 * sizeof(cl_uint), affmGx, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            printf("clEnqueueWriteBuffer");
        }
        err = clEnqueueWriteBuffer(queue, buf_affmGy, CL_TRUE, 0, 8 * sizeof(cl_uint), affmGy, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            printf("clEnqueueWriteBuffer");
        }

        stop_flag = -1;
        err = clEnqueueWriteBuffer(queue, buf_stop_flag, CL_TRUE, 0, sizeof(stop_flag), &stop_flag, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            printf("clEnqueueWriteBuffer\n");
        }
        
        for(int i = 0; i < pre_num; i++){
            read_t2_block(read_file, i, block_size, block_queue, context, queue);
        }

        for(int i = pre_num; i < t2_block_num; ++i){
            if(!stop_thread_flag.load()){
                futures.push_back(pre_pool.enqueue(read_t2_block, std::ref(read_file), i, block_size, std::ref(block_queue), std::ref(context), std::ref(queue)));
            }
        }

        // time start
        auto start = std::chrono::high_resolution_clock::now();

        while(!block_queue.empty()){ // try to prove
            std::unique_lock<std::mutex> lock(queue_mutex);
            cl_mem buf_t2_block = block_queue.front();
            block_queue.pop_front();
            
            secp256k1_fe* t2_block = (secp256k1_fe *)clEnqueueMapBuffer(queue, buf_t2_block, CL_TRUE, CL_MAP_READ, 0, static_cast<size_t>((block_size * 2 + 1) * sizeof(secp256k1_fe)), 0, NULL, NULL, NULL);
            lock.unlock();
            secp256k1_fe* ZinvTree = (secp256k1_fe*)clEnqueueMapBuffer(queue, buf_ZinvTree, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, static_cast<size_t>(Treelen * sizeof(secp256k1_fe)), 0, NULL, NULL, NULL);

            // Ztree leafnode
            std::vector<secp256k1_fe> ZTree(Treelen);
            auto time1 = std::chrono::high_resolution_clock::now();
            
            if(block_size >= ZTree_leafnode_threshold){
                threads_num = ZTree_leafnode_threadsnum[width - 11];
                int chunk_size = (block_size + threads_num - 1) / threads_num;

                for(unsigned int t = 0; t < threads_num; t++){
                    int start = t * chunk_size;
                    int end = std::min(start + chunk_size, block_size);
                    inner_futures.push_back(inner_pool.enqueue(create_ZTree_leafnode, start, end, std::ref(ZTree), t2_block, std::ref(mG_affine)));
                }
                for(auto &future : inner_futures){
                    future.get();
                }
                inner_futures.clear();
            }
            else{
                create_ZTree_leafnode(0, block_size, ZTree, t2_block, mG_affine);
            }
            // Release the t2_block access permission
            clEnqueueUnmapMemObject(queue, buf_t2_block, t2_block, 0, NULL, NULL);

            auto time2 = std::chrono::high_resolution_clock::now();

            // ZTree nonleafnode
            int pre_start = 0, cur_start = 0;
            for(int l = 1; l <= width; l++){
                cur_start += 1 << (width - l + 1);
                int cur_size = 1 << (width - l);
                int cur_end = cur_start + cur_size;

                if(cur_size >= ZTree_nonleafnode_threshold){
                    threads_num = ZTree_nonleafnode_threadsnum[width - l - 10];
                    int chunk_size = (cur_size + threads_num - 1) / threads_num;
                    for(unsigned int t = 0; t < threads_num; t++){
                        int start = cur_start + t * chunk_size;
                        int end = std::min(start + chunk_size, cur_end);
                        inner_futures.push_back(inner_pool.enqueue(create_ZTree_nonleafnode, start, end, pre_start, cur_start, std::ref(ZTree)));
                    }

                    for(auto &future : inner_futures){
                        future.get();
                    }
                    inner_futures.clear();
                }
                else{
                    create_ZTree_nonleafnode(cur_start, cur_end, pre_start, cur_start, ZTree);
                }

                pre_start += 1 << (width - l + 1);
            }

            auto time3 = std::chrono::high_resolution_clock::now();

            // ZinvTree
            pre_start = block_size * 2 - 2, cur_start = block_size * 2 - 2;
            secp256k1_fe_inv(&ZinvTree[cur_start], &ZTree[cur_start]);
            secp256k1_fe_normalize(&ZinvTree[cur_start]);
            for(int l = 1; l <= width; l++){
                pre_start -= 1 << l;
                int cur_size = 1 << (l - 1);
                int cur_end = cur_start + cur_size;

                if(cur_size >= ZinvTree_threshold){
                    threads_num = ZinvTree_threadsnum[l - 1 - 9];
                    int chunk_size = (cur_size + threads_num - 1) / threads_num;
                    for(unsigned int t = 0; t < threads_num; t++){
                        int start = cur_start + t * chunk_size;
                        int end = std::min(start + chunk_size, cur_end);
                        inner_futures.push_back(inner_pool.enqueue(create_ZinvTree, start, end, pre_start, cur_start, std::ref(ZTree), ZinvTree));
                    }

                    for(auto &future : inner_futures){
                        future.get();
                    }
                    inner_futures.clear();
                }
                else{
                    create_ZinvTree(cur_start, cur_end, pre_start, cur_start, ZTree, ZinvTree);
                }
                
                cur_start -= 1 << l;
            }
            auto time4 = std::chrono::high_resolution_clock::now();

            clEnqueueUnmapMemObject(queue, buf_ZinvTree, ZinvTree, 0, NULL, NULL);

            err = clSetKernelArg(traversej_kernel, 5, sizeof(cl_mem), &buf_t2_block);
            if (err != CL_SUCCESS) {
                printf("clSetKernelArg6-6\n");
            }

            auto time5 = std::chrono::high_resolution_clock::now();
            global_size = 1024 * 64;
            local_size = 256;
            err = clEnqueueNDRangeKernel(queue, traversej_kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                printf("clEnqueueNDRangeKernel\n");
            }

            
            err = clEnqueueReadBuffer(queue, buf_stop_flag, CL_TRUE, 0, sizeof(stop_flag), &stop_flag, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                printf("clEnqueueReadBuffer\n");
            }
            clFinish(queue);
            // printf("Queue popped. Current size:%d\n", block_queue.size());
            if(stop_flag != -1){
                // printf("-----success-----\n");
                stop_thread_flag.store(true);
                cv.notify_all();
                err = clEnqueueReadBuffer(queue, buf_m_dec, CL_TRUE, 0, sizeof(m_dec), m_dec, 0, NULL, NULL);
                if (err != CL_SUCCESS) {
                    printf("clEnqueueReadBuffer\n");
                }
                clFinish(queue);
                auto time6 = std::chrono::high_resolution_clock::now();
                sum1 += std::chrono::duration_cast<std::chrono::nanoseconds>(time2 - time1).count();
                sum2 += std::chrono::duration_cast<std::chrono::nanoseconds>(time3 - time2).count();
                sum3 += std::chrono::duration_cast<std::chrono::nanoseconds>(time4 - time3).count();
                sum4 += std::chrono::duration_cast<std::chrono::nanoseconds>(time5 - time4).count();
                sum5 += std::chrono::duration_cast<std::chrono::nanoseconds>(time6 - time5).count();
                sum_dec += std::chrono::duration_cast<std::chrono::nanoseconds>(time6 - time1).count();
                clReleaseMemObject(buf_t2_block);
                break;
            }
            auto time6 = std::chrono::high_resolution_clock::now();
            clReleaseMemObject(buf_t2_block);
            cv.notify_all();

            sum1 += std::chrono::duration_cast<std::chrono::nanoseconds>(time2 - time1).count();
            sum2 += std::chrono::duration_cast<std::chrono::nanoseconds>(time3 - time2).count();
            sum3 += std::chrono::duration_cast<std::chrono::nanoseconds>(time4 - time3).count();
            sum4 += std::chrono::duration_cast<std::chrono::nanoseconds>(time5 - time4).count();
            sum5 += std::chrono::duration_cast<std::chrono::nanoseconds>(time6 - time5).count();
            sum_dec += std::chrono::duration_cast<std::chrono::nanoseconds>(time6 - time1).count();

            if(block_queue.size() == 0){
                printf("size == 0!!!!!!!\n");
                stop_thread_flag.store(true);
            }
        }
        cv.notify_all();
        for(auto &future : futures){
            future.get();
        }
        futures.clear();
        for(auto &buf : block_queue){
            clReleaseMemObject(buf);
        }
        block_queue.clear();

        int equal_flag = 1;
        for (int i = 0; i < 8; i++) {
            if (m_int[i] != m_dec[i]) equal_flag = 0;
        }
        if (equal_flag) {
            success_cnt++;
        }
        // else{
        //     printf("\nwrong value:%lld\n", m_val);
        //     for(int i = 0; i < 8; i++){
        //         printf("%u    %u\n", m_int[i], m_dec[i]);
        //     }
        // }


        // printf("cur ite: %d\n", it + 1);
    }
    double dec_time = sum_dec / static_cast<double>(ite * 1e6);
    double ZTree_leaf_time = sum1 / static_cast<double>(ite * 1e6);
    double ZTree_noneleaf_time = sum2 / static_cast<double>(ite * 1e6);
    double ZinvTree_time = sum3 / static_cast<double>(ite * 1e6);
    double ZinvTree_io_time = sum4 / static_cast<double>(ite * 1e6);
    double traversej_time = sum5 / static_cast<double>(ite * 1e6);

    printf("\n");
    // printf("global_size:    %d\n", global_size);
    // printf("local_size:     %d\n", local_size);
    printf("success_cnt:            %d\n", success_cnt);
    printf("dec_time:               %lf ms\n", dec_time);
    printf("ZTree_leaf_time:        %lf ms\n", ZTree_leaf_time);
    printf("ZTree_noneleaf_time:    %lf ms\n", ZTree_noneleaf_time);
    printf("ZinvTree_time:          %lf ms\n", ZinvTree_time);
    printf("ZinvTree_io_time:       %lf ms\n", ZinvTree_io_time);
    printf("traversej_time:         %lf ms\n", traversej_time);

    // log_t2_chunk_results(Ilen, Jlen, t2_block_num, pre_num, success_cnt, dec_time, ZTree_leaf_time, ZTree_noneleaf_time, ZinvTree_time, ZinvTree_io_time, traversej_time);

    clReleaseKernel(traversej_kernel);
    clReleaseMemObject(buf_table_k);
    clReleaseMemObject(buf_table_v);
    clReleaseMemObject(buf_affmGx);
    clReleaseMemObject(buf_affmGy);
    clReleaseMemObject(buf_ZinvTree);
    clReleaseMemObject(buf_m_dec);
    clReleaseMemObject(buf_stop_flag);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    printf("\ndone\n\n");
    return 0;
}