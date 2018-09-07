import torch.autograd as autograd

class GradReverse(autograd.Function):
    r"""
    Reverse the gradient on the backwards pass. Taken from:
    > https://discuss.pytorch.org/t/solved-reverse-gradients-in-backward-pass/3589/4
    """
    
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

def grad_reverse(x):
    return GradReverse.apply(x)
