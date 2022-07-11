#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>


// CUDA function for simple calculation on any type
template <typename T>
__host__ __device__ T div_round_up(T val, T divisor) {
	return (val + divisor - 1) / divisor;
}


// The real cuda forward_kernel
template <typename scalar_t>
__global__ void forward_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> A,
    const float scale,
    const float bias,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> output) {
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;  // col id
    const uint32_t n = blockIdx.y * blockDim.y + threadIdx.y;  // row id

    if (n < A.size(0) && c < A.size(1)) {  // num block may create some useless thread
        output[n][c] = scale * exp(- A[n][c]) + bias;   // with the help of PackedTensorAccessor
    }
}


/* CUDA instantiate func for scale_exp forward
   @param: A, torch float tensor of (B, N)
   @param: scale, float num
   @param: bias, float num
   @return: output, torch float tensor with the same size as A
*/
torch::Tensor scale_exp_forward_cuda(
    torch::Tensor A, const float scale, const float bias) {
    torch::Tensor output = torch::zeros_like(A);  // space for output

    const uint32_t n_row = A.size(0);  // B
    const uint32_t n_col = A.size(1);  // N
    const dim3 threads(32, 32);  // 2d-block
    const uint32_t thread_per_dim = 32;
    const dim3 blocks(div_round_up(n_col, thread_per_dim), div_round_up(n_row, thread_per_dim));  // 2d-grid

    // instantiate the real executable kernel
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "scale_exp_forward_cuda",  // this will switch actual scalar type
    ([&] {
        forward_kernel<scalar_t><<<blocks, threads>>>(
            A.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            scale, bias,
            output.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
        );
    }));

    return output;
}


// The real cuda backward_kernel
template <typename scalar_t>
__global__ void backward_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> grad_out,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> A,
    const float scale,
    const float bias,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> grad_A) {
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;  // col id
    const uint32_t n = blockIdx.y * blockDim.y + threadIdx.y;  // row id

    if (n < A.size(0) && c < A.size(1)) {  // num block may create some useless thread
        grad_A[n][c] = - scale * exp(-A[n][c]) * grad_out[n][c];   // with the help of PackedTensorAccessor
    }
}


/* CUDA instantiate func for scale_exp backward
   @param: grad_out, torch float tensor of (B, N), final grad
   @param: A, torch float tensor of (B, N)
   @param: scale, float num
   @param: bias, float num
   @return: grad_A, torch float tensor with the same size as A
*/
torch::Tensor scale_exp_backward_cuda(
    torch::Tensor grad_out, torch::Tensor A, const float scale, const float bias) {
    torch::Tensor grad_A = torch::zeros_like(A);  // space for output

    const uint32_t n_row = A.size(0);  // B
    const uint32_t n_col = A.size(1);  // N
    const dim3 threads(32, 32);  // 2d-block
    const uint32_t thread_per_dim = 32;
    const dim3 blocks(div_round_up(n_col, thread_per_dim), div_round_up(n_row, thread_per_dim));  // 2d-grid

    // instantiate the real executable kernel
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "scale_exp_backward_cuda",  // this will switch actual scalar type
    ([&] {
        backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_out.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            A.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            scale, bias,
            grad_A.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
        );
    }));

    return grad_A;
}
