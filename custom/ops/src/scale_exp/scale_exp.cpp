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
torch::Tensor scale_exp_forward_cuda(torch::Tensor A, const float scale, const float bias);


/* c++ wrapper of scale_exp forward func
   py: scale_exp_forward(A, scale, bias)
   @param: A, torch float tensor of (B, N)
   @param: scale, float num
   @param: bias, float num
   @return: output, torch float tensor with the same size as A
*/
torch::Tensor scale_exp_forward(torch::Tensor A, const float scale, const float bias) {
    //checking
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
   @param: grad, torch float tensor of (B, N), final grad
   @param: A, torch float tensor of (B, N)
   @param: scale, float num
   @param: bias, float num
   @return: output, torch float tensor with the same size as A
*/
torch::Tensor scale_exp_backward(
    torch::Tensor grad_out,
    torch::Tensor A,
    const float scale,
    const float bias) {
    //checking
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
