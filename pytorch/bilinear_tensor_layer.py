import torch

class BilinearTensorFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vec1, vec2, tensor_weights, linear_weights1, linear_weights2, bias_weights):
        ctx.save_for_backward(vec1, vec2, tensor_weights, linear_weights1, linear_weights2, bias_weights)

        # Bilinear tensor product.
        bilinear = torch.matmul(torch.matmul(vec1, tensor_weights).squeeze(), vec2.t()).t()

        # Linear output.
        linear1 = torch.matmul(linear_weights1, vec1.t()).t()
        linear2 = torch.matmul(linear_weights2, vec2.t()).t()

        # Bias output.
        return bilinear + linear1 + linear2 + bias_weights

    def backward(ctx, grad_output):
        # Retrieve the values stored during feedforward phase.
        vec1, vec2, tensor_weights, linear_weights1, linear_weights2, bias_weights = ctx.saved_variables

        # Initialize gradient to None.
        grad_vec1 = grad_vec2 = grad_tensor = grad_linear1 = grad_linear2 = grad_bias = None

        # Input vector gradients.
        if ctx.needs_input_grad[0]:
            grad_vec1 = grad_output.matmul(linear_weights1) + tensor_weights * grad_output.squeeze().unsqueeze(-1).unsqueeze(-1).expand(k, n, n)
            torch.matmul(vec2, torch.matmul(tensor_weights, grad_output.t()).squeeze())

            grad_vec1 = grad_output.matmul(linear_weights1) + torch.matmul(vec2, torch.matmul(tensor_weights, grad_output.t()).squeeze())

        if ctx.needs_input_grad[1]:
            grad_vec2 = grad_output.matmul(linear_weights2) + torch.matmul(grad_output, torch.matmul(vec1, tensor_weights).squeeze())

        # Tensor weight gradients.
        if ctx.needs_input_grad[2]:
            grad_tensor = torch.matmul(torch.matmul(vec1.t(), vec2).unsqueeze(2), grad_output)

        # Linear weights.
        if ctx.needs_input_grad[3]:
            grad_linear1 = grad_output.t().matmul(vec1)

        if ctx.needs_input_grad[4]:
            grad_linear2 = grad_output.t().matmul(vec2)

        # Bias weights.
        if ctx.needs_input_grad[5]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_vec1, grad_vec2, grad_tensor, grad_linear1, grad_linear2, grad_bias


class BilinearTensorLayer(torch.nn.Module):
    def __init__(self, input_vec_dim, output_vec_dim):
        super(BilinearTensorLayer, self).__init__()
        self.input_vec_dim = input_vec_dim
        self.output_vec_dim = output_vec_dim

        self.tensor_weights = nn.Parameter(torch.Tensor(output_vec_dim, input_vec_dim, input_vec_dim))
        self.linear_weights1 = nn.Parameter(torch.Tensor(output_vec_dim, input_vec_dim))
        self.linear_weights2 = nn.Parameter(torch.Tensor(output_vec_dim, input_vec_dim))
        self.bias_weights = nn.Parameter(torch.Tensor(output_vec_dim))

        # Random weight initialization.
        self.tensor_weights.data.normal_(0.0, 1.0 / (input_vec_dim * input_vec_dim * output_vec_dim))
        self.linear_weights1.data.normal_(0.0, 1.0 / (output_vec_dim * input_vec_dim))
        self.linear_weights2.data.normal_(0.0, 1.0 / (output_vec_dim * input_vec_dim))
        self.bias_weights.data.fill_(0.0)

    def forward(self, vec1, vec2):
        return BilinearTensorFunction.apply(vec1, vec2, self.tensor_weights, self.linear_weights1, self.linear_weights2, self.bias_weights)