// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: leoyluo@tencent.com (Yue Luo)
//
// helper func

#ifdef HELPER_H
#define HELPER_H

// CUDA function for simple calculation on any type
template <typename T>
inline __host__ __device__ T div_round_up(T val, T divisor) {
    return (val + divisor - 1) / divisor;
}

// Just a simple cuda inline function
template <typename T>
inline __host__ __device__ T identity(T z) {
    return z;
}

#endif
