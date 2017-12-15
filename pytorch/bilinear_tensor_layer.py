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

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve the values stored during feedforward phase.
        vec1, vec2, tensor_weights, linear_weights1, linear_weights2, bias_weights = ctx.saved_variables
        print(grad_output)

        # Initialize gradient to None.
        grad_vec1 = grad_vec2 = grad_tensor = grad_linear1 = grad_linear2 = grad_bias = None

        # Input vector gradients.
        if ctx.needs_input_grad[0]:
            grad_vec1 = grad_output.matmul(linear_weights1)

            tprod = torch.matmul(tensor_weights, vec2)
            for si in range(tensor_weights.size()[0]):
                tprod[si] *= grad_output.t()[0]

            grad_vec1 += torch.sum(tprod, 0).t()

        if ctx.needs_input_grad[1]:
            grad_vec2 = grad_output.matmul(linear_weights2)

            tprod = torch.matmul(vec1, tensor_weights)
            for si in range(tensor_weights.size()[0]):
                tprod[si] *= grad_output.t()[0]

            grad_vec2 += torch.sum(tprod, 0)

        # Tensor weight gradients.
        if ctx.needs_input_grad[2]:
            grad_tensor = torch.matmul(vec1.t(), vec2).unsqueeze(0).expand(tensor_weights.size()[0], tensor_weights.size()[1], tensor_weights.size()[2])

            tprod = torch.matmul(tensor_weights, vec2)
            for si in range(tensor_weights.size()[0]):
                grad_tensor[si] *= grad_output.t()[0]

        # Linear weights.
        if ctx.needs_input_grad[3]:
            grad_linear1 = torch.matmul(grad_output, vec1)

        if ctx.needs_input_grad[4]:
            grad_linear2 = torch.matmul(grad_output, vec2)

        # Bias weights.
        if ctx.needs_input_grad[5]:
            grad_bias = grad_output

        return grad_vec1, grad_vec2, grad_tensor, grad_linear1, grad_linear2, grad_bias


class BilinearTensorLayer(torch.nn.Module):
    def __init__(self, input_vec_dim, output_vec_dim):
        super(BilinearTensorLayer, self).__init__()
        self.input_vec_dim = input_vec_dim
        self.output_vec_dim = output_vec_dim

        # Set up tensor, linear, and bias weights.
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
        # Bilinear tensor product.
        bilinear = torch.matmul(torch.matmul(vec1, self.tensor_weights).squeeze(), vec2.t()).t()

        # Linear output.
        linear1 = torch.matmul(self.linear_weights1, vec1.t()).t()
        linear2 = torch.matmul(self.linear_weights2, vec2.t()).t()

        # Add in the bias vector also.
        return bilinear + linear1 + linear2 + self.bias_weights