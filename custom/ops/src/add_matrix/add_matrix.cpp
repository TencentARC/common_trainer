#include <torch/extension.h>
#include <torch/torch.h>


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor")
#define CHECK_IS_FLOATING(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float || \
                                         x.scalar_type() == at::ScalarType::Half || \
                                         x.scalar_type() == at::ScalarType::Double, \
                                         #x " must be a floating tensor")


// define the real cuda function to be called by c++ wrapper.
torch::Tensor add_matrix_forward_cuda(torch::Tensor A, torch::Tensor B);


/* c++ wrapper of add_matrix forward func
   py: add_matrix_forward(A, B)
   @param: A, torch float tensor of (B, N)
   @param: B, torch float tensor with same size as B
   @return: output, torch float tensor with the same size as A
*/
torch::Tensor add_matrix_forward(torch::Tensor A, torch::Tensor B) {
    //checking
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
    //checking
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
