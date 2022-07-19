#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>


// CUDA function for simple calculation on any type
template <typename T>
__host__ __device__ T div_round_up(T val, T divisor) {
	return (val + divisor - 1) / divisor;
}

// Just a simple cuda inline function
template <typename scalar_t>
__device__ __forceinline__ scalar_t identity(scalar_t z) {
    return z;
}


// The real cuda forward_kernel
template <typename scalar_t>
__global__ void forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output) {
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;  // col id
    const uint32_t n = blockIdx.y * blockDim.y + threadIdx.y;  // row id

    if (n < A.size(0) && c < A.size(1)) {  // num block may create some useless thread
        output[n][c] = identity(A[n][c] + B[n][c]);   // with the help of PackedTensorAccessor32
    }
}


/* CUDA instantiate func for add_matrix
   @param: A, torch float tensor of (B, N)
   @param: B, torch float tensor with same size as B
   @return: output, torch float tensor with the same size as A
*/
torch::Tensor add_matrix_forward_cuda(torch::Tensor A, torch::Tensor B) {
    torch::Tensor output = torch::zeros_like(A);  // space for output

    const uint32_t n_row = A.size(0);  // B
    const uint32_t n_col = A.size(1);  // N
    const dim3 threads(32, 32);  // 2d-block
    const uint32_t thread_per_dim = 32;
    const dim3 blocks(div_round_up(n_col, thread_per_dim), div_round_up(n_row, thread_per_dim));  // 2d-grid

    // instantiate the real executable kernel
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "add_matrix_forward_cuda",  // this will switch actual scalar type
    ([&] {
        forward_kernel<scalar_t><<<blocks, threads>>>(
            A.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            B.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()
        );
    }));

    return output;
}
