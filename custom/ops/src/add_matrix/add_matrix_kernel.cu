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
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> A,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> B,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> output) {
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t n = blockIdx.y;

    if (c < A.size(1)) {  // num block may create some useless thread
        output[n][c] = identity(A[n][c] + B[n][c]);   // with the help of PackedTensorAccessor
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
    const uint32_t threads = 1024;  // You should set the num_threads for parallelization
    const dim3 blocks(div_round_up(n_col, threads), n_row);  // 2-dim block to cover all elements
    // const int blocks = div_round_up(n_row * n_col, threads); // 1-dim block is also allowed, but not easy to fine idx

    // instantiate the real executable kernel
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "add_matrix_forward_cuda",  // this will switch actual scalar type
    ([&] {
        forward_kernel<scalar_t><<<blocks, threads>>>(
            A.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            B.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            output.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
        );
    }));

    return output;
}
