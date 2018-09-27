import torch.autograd as autograd


class GradReverse(autograd.Function):
    r"""
    Reverse the gradient on the backwards pass.
    """

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


def grad_reverse(x):
    return GradReverse.apply(x)
