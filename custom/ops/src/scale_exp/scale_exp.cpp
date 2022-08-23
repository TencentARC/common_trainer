// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: leoyluo@tencent.com (Yue Luo)
//
// transformation of one tensor, C = s * exp(-A) + b

#include <torch/torch.h>

#include "utils.h"


// define the real cuda function to be called by c++ wrapper.
torch::Tensor scale_exp_forward_cuda(torch::Tensor A, const float scale, const float bias);


/* c++ wrapper of scale_exp forward func
   py: scale_exp_forward(A, scale, bias)
   @param: A, torch float tensor of (B, N)
   @param: scale, float num
   @param: bias, float num
   @return: output, torch float tensor with the same size as A
*/
torch::Tensor scale_exp_forward(torch::Tensor A, const float scale, const float bias) {
    // checking
    CHECK_INPUT(A)
    CHECK_IS_FLOATING(A)

    // call actual cuda function
    return scale_exp_forward_cuda(A, scale, bias);
}


// define the real cuda function to be called by c++ wrapper.
torch::Tensor scale_exp_backward_cuda(
    torch::Tensor grad_out, torch::Tensor A, const float scale, const float bias);


/* c++ wrapper of add_matrix backward func
   py: scale_exp_backward(grad, A, scale, bias)
   @param: grad_out, torch float tensor of (B, N), final grad
   @param: A, torch float tensor of (B, N)
   @param: scale, float num
   @param: bias, float num
   @return: grad_A, torch float tensor with the same size as A
*/
torch::Tensor scale_exp_backward(
    torch::Tensor grad_out,
    torch::Tensor A,
    const float scale,
    const float bias) {
    // checking
    CHECK_INPUT(A)
    CHECK_INPUT(grad_out)

    CHECK_IS_FLOATING(A)
    CHECK_IS_FLOATING(grad_out)

    // call actual cuda function
    return scale_exp_backward_cuda(grad_out, A, scale, bias);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scale_exp_forward", &scale_exp_forward, "Scale exp forward (CUDA)");
    m.def("scale_exp_backward", &scale_exp_backward, "Scale exp backward (CUDA)");
}
