import torch
from torch import nn
from torch.autograd import Function

# Esto debería estar en la librería de Sparse, es un TODO
def kl_divergence_gradient(p: float, q: torch.Tensor, apply_sigmoid = True) -> torch.Tensor:
    r'''
        Compute the gradient of the KL divergence with respect to the input tensor.

        Parameters
        ----------
            p: float
                Sparsity parameter, typically a small value close to zero (say p=0.05).
            
            q: torch.Tensor, shape (batch_size, n_features)
                Input tensor which will be penalized deviating significantly from p.
    '''
    # check if tensor belong to a convolutional output or not
    dim = 2 if len(q.shape) == 4 else 1

    q = torch.sigmoid(q) if apply_sigmoid else q # sigmoid because we need the probability distributions
    q_grad = q * (1-q) if apply_sigmoid else 1

    rho_hat = q.flatten(dim)
    rho = torch.ones_like(rho_hat).to(q.device) * p

    eps = 1e-12
    kl_grad = -(rho_hat - rho)/((rho_hat-1)*rho_hat + eps)

    return q_grad * kl_grad 

class SparseWeightedFeatureSelectorFunction(Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor, sparse_rate: float) -> torch.Tensor:
        ctx.save_for_backward(input, weight, sparse_rate)
        output = input * weight
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, sparse_rate = ctx.saved_tensors
        grad_input = grad_weight = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output * weight
            # print('Grad. Input: {}'.format(grad_input))
        if ctx.needs_input_grad[1]:
            grad_weight = (grad_output * input).sum(dim=0).reshape(weight.shape) # Dot product
            sparse_grad = (kl_divergence_gradient(0.05, input * weight, apply_sigmoid=True) * input).sum(dim=0).reshape(weight.shape)
            grad_weight += sparse_rate*sparse_grad
            # print('Grad. Weight: {}'.format(grad_weight))

        return grad_input, grad_weight, None

class SparseWeightedFeatureSelector(nn.Module):
    r'''
        Feature Selector by a Sparse Weighted method. 

        Sparsity by KL Divergence of a Binomal distribution.
        Reference: https://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf
    '''
    def __init__(self, n_features:int, sparse_rate: float):
        super(SparseWeightedFeatureSelector, self).__init__()  
        self.sparse_rate = nn.Parameter(torch.tensor(sparse_rate), requires_grad = False)
        self.weight = nn.Parameter(torch.rand((n_features,), requires_grad=True))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return SparseWeightedFeatureSelectorFunction.apply(input, self.weight, self.sparse_rate)