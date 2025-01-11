#include "ecc.cl"

typedef struct {
    /* A field element f represents the sum(i=0..4, f.n[i] << (i*52)) mod p,
     * where p is the field modulus, 2^256 - 2^32 - 977.
     *
     * The individual limbs f.n[i] can exceed 2^52; the field's magnitude roughly
     * corresponds to how much excess is allowed. The value
     * sum(i=0..4, f.n[i] << (i*52)) may exceed p, unless the field element is
     * normalized. */
    ulong n[5];
    /*
     * Magnitude m requires:
     *     n[i] <= 2 * m * (2^52 - 1) for i=0..3
     *     n[4] <= 2 * m * (2^48 - 1)
     *
     * Normalized requires:
     *     n[i] <= (2^52 - 1) for i=0..3
     *     sum(i=0..4, n[i] << (i*52)) < p
     *     (together these imply n[4] <= 2^48 - 1)
     */
    //SECP256K1_FE_VERIFY_FIELDS
} secp256k1_fe;

void convert_l5_to_i8(__global ulong n[5], uint r[8]) {
    r[0] = n[0] & 0xFFFFFFFF;
    r[1] = ((n[1] & 0xFFF) << 20) | (n[0] >> 32);
    r[2] = (n[1] >> 12) & 0xFFFFFFFF;
    r[3] = ((n[2] & 0xFFFFFF) << 8) | (n[1] >> 44);
    r[4] = ((n[3] & 0xF) << 28) | (n[2] >> 24);
    r[5] = (n[3] >> 4) & 0xFFFFFFFF;
    r[6] = ((n[4] & 0xFFFF) << 16) | (n[3] >> 36);
    r[7] = n[4] >> 16;
}

__kernel void Traversej(
    const int block_size,
    const int Ilen,
    __global secp256k1_fe* ZinvTree,
    __global uint* affmGx,
    __global uint* affmGy,
    __global secp256k1_fe* t2,
    __global uint* m,
    __global int* stop_flag, 
    __global uint* table_k,
    __global int* table_v
) {
    int Imax = 1 << Ilen;
    uint cuckoolen = Imax * 1.3;
    int gid = get_global_id(0);
    int step = get_global_size(0);

    int res_v;
    int flag;
    int check_flag;
    uint res_v_array[8];

    struct secp256k1 g_tmp;
    set_precomputed_basepoint_g(&g_tmp);

    // presearch
    if (*stop_flag != -1) {
        return;
    }
    res_v = 0;
    flag = 0;
    for (int hash_id = 1; hash_id <= 3; hash_id++) {
        uint x = affmGx[8 - hash_id * 2];
        uint x_key = x;
        uint h = affmGx[8 - hash_id * 2 + 1] % cuckoolen;
        if (table_k[h] == x_key) {
            res_v = table_v[h];
            flag = 1;
            break;
        }
    }
    if (flag == 1) {
        res_v_array[0] = res_v;
        res_v_array[1] = 0;
        res_v_array[2] = 0;
        res_v_array[3] = 0;
        res_v_array[4] = 0;
        res_v_array[5] = 0;
        res_v_array[6] = 0;
        res_v_array[7] = 0;

        uint mGx_pre_tmp[8];
        uint mGy_pre_tmp[8];
        point_mul_xy(mGx_pre_tmp, mGy_pre_tmp, res_v_array, &g_tmp); // check start
        check_flag = 1;
        for (int i = 0; i < 8; i++) {
            if (mGx_pre_tmp[i] != affmGx[i] || mGy_pre_tmp[i] != affmGy[i]) {
                check_flag = 0;
            }
        }
        if (check_flag == 1) {
            m[0] = res_v;
            *stop_flag = gid;
            return;
        }
        else {
            m[0] = SECP256K1_P0 - res_v;
            m[1] = SECP256K1_P1;
            m[2] = SECP256K1_P2;
            m[3] = SECP256K1_P3;
            m[4] = SECP256K1_P4;
            m[5] = SECP256K1_P5;
            m[6] = SECP256K1_P6;
            m[7] = SECP256K1_P7;
            *stop_flag = gid;
            return;
        }
    }

    uint Qx[8];
    uint Qxinv[8];
    uint p[8] = { SECP256K1_P0, SECP256K1_P1, SECP256K1_P2, SECP256K1_P3, SECP256K1_P4, SECP256K1_P5, SECP256K1_P6, SECP256K1_P7 };

    int st = 0;
    uint block_index[8];
    convert_l5_to_i8(t2[0].n, block_index);
    int j_start = block_index[0] * block_size;
    for (; st <= block_size - step; st += step) {
        if (*stop_flag != -1) {
            return;
        }

        int t2_index = st + gid + 1;
        int j = t2_index + j_start;

        uint k[8];

        uint t2jx[8];
        convert_l5_to_i8(t2[t2_index * 2].n, t2jx);
        uint t2jy[8];
        convert_l5_to_i8(t2[t2_index * 2 + 1].n, t2jy);
        uint inv[8];
        convert_l5_to_i8(ZinvTree[t2_index - 1].n, inv);

        add_mod(k, affmGx, t2jx);
        uint mGx_tmp[8];
        uint mGy_tmp[8];
        uint m1[8];
        uint m2[8];

        // Qx start
        // Qx = mG + j(2ImaxG)
        sub_mod(Qx, affmGy, t2jy);
        mul_mod(Qx, Qx, inv);
        mul_mod(Qx, Qx, Qx);
        sub_mod(Qx, Qx, k);
        // Search(Ilen, Qx, table_k, table_v, res_v, flag);
        res_v = 0;
        flag = 0;
        for (int hash_id = 1; hash_id <= 3; hash_id++) {
            uint x = Qx[8 - hash_id * 2];
            uint x_key = x;
            uint h = Qx[8 - hash_id * 2 + 1] % cuckoolen;
            if (table_k[h] == x_key) {
                res_v = table_v[h];
                flag = 1;
                break;
            }
        }

        if (flag == 1) {
            res_v_array[0] = res_v;
            res_v_array[1] = 0;
            res_v_array[2] = 0;
            res_v_array[3] = 0;
            res_v_array[4] = 0;
            res_v_array[5] = 0;
            res_v_array[6] = 0;
            res_v_array[7] = 0;

            uint m_base[8] = { 0 };
            m_base[0] = j * 2;
            sub_mod(m_base, p, m_base);
            uint a[8] = { 0 };
            a[0] = Imax;
            mul_mod(m_base, m_base, a); // m_base = -j * Imax * 2

            add_mod(m1, m_base, res_v_array); // m1 = -j * imax * 2 + i
            point_mul_xy(mGx_tmp, mGy_tmp, m1, &g_tmp); // check start
            check_flag = 1;
            for (int i = 0; i < 8; i++) {
                if (mGx_tmp[i] != affmGx[i] || mGy_tmp[i] != affmGy[i]) {
                    check_flag = 0;
                }
            }
            if (check_flag == 1) {
                m[0] = m1[0];
                m[1] = m1[1];
                m[2] = m1[2];
                m[3] = m1[3];
                m[4] = m1[4];
                m[5] = m1[5];
                m[6] = m1[6];
                m[7] = m1[7];
                *stop_flag = gid;
                return;
            }

            sub_mod(m2, m_base, res_v_array); // m2 = -j * Imax * 2 - i
            point_mul_xy(mGx_tmp, mGy_tmp, m2, &g_tmp); // check start
            check_flag = 1;
            for (int i = 0; i < 8; i++) {
                if (mGx_tmp[i] != affmGx[i] || mGy_tmp[i] != affmGy[i]) {
                    check_flag = 0;
                }
            }
            if (check_flag == 1) {
                m[0] = m2[0];
                m[1] = m2[1];
                m[2] = m2[2];
                m[3] = m2[3];
                m[4] = m2[4];
                m[5] = m2[5];
                m[6] = m2[6];
                m[7] = m2[7];
                *stop_flag = gid;
                return;
            }
        }

        //Qxinv start

        sub_mod(t2jy, p, t2jy);
        sub_mod(Qxinv, affmGy, t2jy);
        mul_mod(Qxinv, Qxinv, inv);
        mul_mod(Qxinv, Qxinv, Qxinv);
        sub_mod(Qxinv, Qxinv, k);

        // Search(Ilen, Qxinv, table_k, table_v, res_v, flag);
        res_v = 0;
        flag = 0;
        for (int hash_id = 1; hash_id <= 3; hash_id++) {
            uint x = Qxinv[8 - hash_id * 2];
            uint x_key = x;
            uint h = Qxinv[8 - hash_id * 2 + 1] % cuckoolen;
            if (table_k[h] == x_key) {
                res_v = table_v[h];
                flag = 1;
                break;
            }
        }

        if (flag == 1) {
            res_v_array[0] = res_v;
            res_v_array[1] = 0;
            res_v_array[2] = 0;
            res_v_array[3] = 0;
            res_v_array[4] = 0;
            res_v_array[5] = 0;
            res_v_array[6] = 0;
            res_v_array[7] = 0;

            uint m_base_inv[8] = { 0 };
            m_base_inv[0] = j * 2;
            uint b[8] = { 0 };
            b[0] = Imax;
            mul_mod(m_base_inv, m_base_inv, b); // m_base_inv = j * Imax * 2

            add_mod(m1, m_base_inv, res_v_array); // m1 = j * Imax * 2 + i
            point_mul_xy(mGx_tmp, mGy_tmp, m1, &g_tmp); // check start
            check_flag = 1;
            for (int i = 0; i < 8; i++) {
                if (mGx_tmp[i] != affmGx[i] || mGy_tmp[i] != affmGy[i]) {
                    check_flag = 0;
                }
            }
            if (check_flag == 1) {
                m[0] = m1[0];
                m[1] = m1[1];
                m[2] = m1[2];
                m[3] = m1[3];
                m[4] = m1[4];
                m[5] = m1[5];
                m[6] = m1[6];
                m[7] = m1[7];
                *stop_flag = gid;
                return;
            }

            sub_mod(m2, m_base_inv, res_v_array); // m2 = j * Imax * 2 - i
            point_mul_xy(mGx_tmp, mGy_tmp, m2, &g_tmp);
            check_flag = 1;
            for (int i = 0; i < 8; i++) {
                if (mGx_tmp[i] != affmGx[i] || mGy_tmp[i] != affmGy[i]) {
                    check_flag = 0;
                }
            }
            if (check_flag == 1) {
                m[0] = m2[0];
                m[1] = m2[1];
                m[2] = m2[2];
                m[3] = m2[3];
                m[4] = m2[4];
                m[5] = m2[5];
                m[6] = m2[6];
                m[7] = m2[7];
                *stop_flag = gid;
                return;
            }
        }
    }
    if (st != block_size) {
        if (gid < block_size - st) {
            if (*stop_flag != -1) {
                return;
            }

            int t2_index = st + gid + 1;
            int j = t2_index + j_start;

            uint k[8];

            uint t2jx[8];
            convert_l5_to_i8(t2[t2_index * 2].n, t2jx);
            uint t2jy[8];
            convert_l5_to_i8(t2[t2_index * 2 + 1].n, t2jy);
            uint inv[8];
            convert_l5_to_i8(ZinvTree[t2_index - 1].n, inv);

            add_mod(k, affmGx, t2jx);
            uint mGx_tmp[8];
            uint mGy_tmp[8];
            uint m1[8];
            uint m2[8];

            // Qx start
            // Qx = mG + j(2ImaxG)
            sub_mod(Qx, affmGy, t2jy);
            mul_mod(Qx, Qx, inv);
            mul_mod(Qx, Qx, Qx);
            sub_mod(Qx, Qx, k);
            // Search(Ilen, Qx, table_k, table_v, res_v, flag);
            res_v = 0;
            flag = 0;
            for (int hash_id = 1; hash_id <= 3; hash_id++) {
                uint x = Qx[8 - hash_id * 2];
                uint x_key = x;
                uint h = Qx[8 - hash_id * 2 + 1] % cuckoolen;
                if (table_k[h] == x_key) {
                    res_v = table_v[h];
                    flag = 1;
                    break;
                }
            }

            if (flag == 1) {
                res_v_array[0] = res_v;
                res_v_array[1] = 0;
                res_v_array[2] = 0;
                res_v_array[3] = 0;
                res_v_array[4] = 0;
                res_v_array[5] = 0;
                res_v_array[6] = 0;
                res_v_array[7] = 0;

                uint m_base[8] = { 0 };
                m_base[0] = j * 2;
                sub_mod(m_base, p, m_base);
                uint a[8] = { 0 };
                a[0] = Imax;
                mul_mod(m_base, m_base, a); // m_base = -j * Imax * 2

                add_mod(m1, m_base, res_v_array); // m1 = -j * imax * 2 + i
                point_mul_xy(mGx_tmp, mGy_tmp, m1, &g_tmp); // check start
                check_flag = 1;
                for (int i = 0; i < 8; i++) {
                    if (mGx_tmp[i] != affmGx[i] || mGy_tmp[i] != affmGy[i]) {
                        check_flag = 0;
                    }
                }
                if (check_flag == 1) {
                    m[0] = m1[0];
                    m[1] = m1[1];
                    m[2] = m1[2];
                    m[3] = m1[3];
                    m[4] = m1[4];
                    m[5] = m1[5];
                    m[6] = m1[6];
                    m[7] = m1[7];
                    *stop_flag = gid;
                    return;
                }

                sub_mod(m2, m_base, res_v_array); // m2 = -j * Imax * 2 - i
                point_mul_xy(mGx_tmp, mGy_tmp, m2, &g_tmp); // check start
                check_flag = 1;
                for (int i = 0; i < 8; i++) {
                    if (mGx_tmp[i] != affmGx[i] || mGy_tmp[i] != affmGy[i]) {
                        check_flag = 0;
                    }
                }
                if (check_flag == 1) {
                    m[0] = m2[0];
                    m[1] = m2[1];
                    m[2] = m2[2];
                    m[3] = m2[3];
                    m[4] = m2[4];
                    m[5] = m2[5];
                    m[6] = m2[6];
                    m[7] = m2[7];
                    *stop_flag = gid;
                    return;
                }
            }

            //Qxinv start

            sub_mod(t2jy, p, t2jy);
            sub_mod(Qxinv, affmGy, t2jy);
            mul_mod(Qxinv, Qxinv, inv);
            mul_mod(Qxinv, Qxinv, Qxinv);
            sub_mod(Qxinv, Qxinv, k);

            // Search(Ilen, Qxinv, table_k, table_v, res_v, flag);
            res_v = 0;
            flag = 0;
            for (int hash_id = 1; hash_id <= 3; hash_id++) {
                uint x = Qxinv[8 - hash_id * 2];
                uint x_key = x;
                uint h = Qxinv[8 - hash_id * 2 + 1] % cuckoolen;
                if (table_k[h] == x_key) {
                    res_v = table_v[h];
                    flag = 1;
                    break;
                }
            }

            if (flag == 1) {
                res_v_array[0] = res_v;
                res_v_array[1] = 0;
                res_v_array[2] = 0;
                res_v_array[3] = 0;
                res_v_array[4] = 0;
                res_v_array[5] = 0;
                res_v_array[6] = 0;
                res_v_array[7] = 0;

                uint m_base_inv[8] = { 0 };
                m_base_inv[0] = j * 2;
                uint b[8] = { 0 };
                b[0] = Imax;
                mul_mod(m_base_inv, m_base_inv, b); // m_base_inv = j * Imax * 2

                add_mod(m1, m_base_inv, res_v_array); // m1 = j * Imax * 2 + i
                point_mul_xy(mGx_tmp, mGy_tmp, m1, &g_tmp); // check start
                check_flag = 1;
                for (int i = 0; i < 8; i++) {
                    if (mGx_tmp[i] != affmGx[i] || mGy_tmp[i] != affmGy[i]) {
                        check_flag = 0;
                    }
                }
                if (check_flag == 1) {
                    m[0] = m1[0];
                    m[1] = m1[1];
                    m[2] = m1[2];
                    m[3] = m1[3];
                    m[4] = m1[4];
                    m[5] = m1[5];
                    m[6] = m1[6];
                    m[7] = m1[7];
                    *stop_flag = gid;
                    return;
                }

                sub_mod(m2, m_base_inv, res_v_array); // m2 = -j * Imax * 2 - i
                point_mul_xy(mGx_tmp, mGy_tmp, m2, &g_tmp);
                check_flag = 1;
                for (int i = 0; i < 8; i++) {
                    if (mGx_tmp[i] != affmGx[i] || mGy_tmp[i] != affmGy[i]) {
                        check_flag = 0;
                    }
                }
                if (check_flag == 1) {
                    m[0] = m2[0];
                    m[1] = m2[1];
                    m[2] = m2[2];
                    m[3] = m2[3];
                    m[4] = m2[4];
                    m[5] = m2[5];
                    m[6] = m2[6];
                    m[7] = m2[7];
                    *stop_flag = gid;
                    return;
                }
            }
        }
    }
}