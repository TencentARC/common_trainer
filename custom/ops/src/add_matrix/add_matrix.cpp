// Copyright 2022 Tencent Inc. All rights reserved.
//
// Author: leoyluo@tencent.com (Yue Luo)
//
// add two tensor, C = A + B

#include <torch/torch.h>

#include "utils.h"


// define the real cuda function to be called by c++ wrapper.
torch::Tensor add_matrix_forward_cuda(torch::Tensor A, torch::Tensor B);


/* c++ wrapper of add_matrix forward func
   py: add_matrix_forward(A, B)
   @param: A, torch float tensor of (B, N)
   @param: B, torch float tensor with same size as B
   @return: output, torch float tensor with the same size as A
*/
torch::Tensor add_matrix_forward(torch::Tensor A, torch::Tensor B) {
    // checking
    CHECK_INPUT(A)
    CHECK_INPUT(B)

    CHECK_IS_FLOATING(A)
    CHECK_IS_FLOATING(B)

    // call actual cuda function
    return add_matrix_forward_cuda(A, B);
}


/* c++ wrapper of add_matrix backward func
   py: add_matrix_backward(grad, A, B), actually A,B is useless, just for sample
   @param: grad_out, torch float tensor of (B, N), final grad
   @param: A, torch float tensor of (B, N)
   @param: B, torch float tensor with same size as B
   @return: grad_A, grad_B, vector of torch float tensor with the same size as A, B
*/
std::vector<torch::Tensor> add_matrix_backward(torch::Tensor grad_out, torch::Tensor A, torch::Tensor B) {
    // checking
    CHECK_INPUT(A)
    CHECK_INPUT(B)
    CHECK_INPUT(grad_out)

    CHECK_IS_FLOATING(A)
    CHECK_IS_FLOATING(B)
    CHECK_IS_FLOATING(grad_out)

    return {grad_out, grad_out};  // grad is simply dy*1
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_matrix_forward", &add_matrix_forward, "Add matrix forward (CUDA)");
    m.def("add_matrix_backward", &add_matrix_backward, "Add matrix backward (CUDA)");
}
