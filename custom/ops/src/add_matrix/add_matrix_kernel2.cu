// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: leoyluo@tencent.com (Yue Luo)
//
// add two tensor, C = A + B

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "helper.h"


// The real cuda forward_kernel. This use the ptr version, but does seems to be faster.
template <typename scalar_t>
__global__ void forward_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ output,
    const uint32_t n_row,
    const uint32_t n_col) {
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;  // col id
    const uint32_t n = blockIdx.y * blockDim.y + threadIdx.y;  // row id

    if (n < n_row && c < n_col) {  // num block may create some useless thread
        output[n*n_col+c] = identity(A[n*n_col+c] + B[n*n_col+c]);
        return;
    }
}


/* CUDA instantiate func for add_matrix
   @param: A, torch float tensor of (B, N)
   @param: B, torch float tensor with same size as B
   @return: output, torch float tensor with the same size as A
*/
torch::Tensor add_matrix_forward_cuda(torch::Tensor A, torch::Tensor B) {
    at::cuda::CUDAGuard device_guard(A.device());
    torch::Tensor output = torch::zeros_like(A);  // space for output

    const uint32_t n_row = A.size(0);  // B
    const uint32_t n_col = A.size(1);  // N
    const uint32_t thread_per_dim = 32;
    const dim3 threads(thread_per_dim, thread_per_dim);  // 2d-block
    const dim3 blocks(div_round_up(n_col, thread_per_dim), div_round_up(n_row, thread_per_dim));  // 2d-grid

    // instantiate the real executable kernel
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "add_matrix_forward_cuda",  // this will switch actual scalar type
    ([&] {
        forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            A.data<scalar_t>(),
            B.data<scalar_t>(),
            output.data<scalar_t>(),
            n_row,
            n_col
        );
    }));

    AT_CUDA_CHECK(cudaGetLastError());

    return output;
}
