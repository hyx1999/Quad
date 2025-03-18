import torch

def flatten_last_dim_and_return_shape(x: torch.Tensor | None):
    if x is None:
        return None, None
    else:
        shape_excl_last = x.shape[:-1]
        x = x.view(-1, x.shape[-1])
        return x, shape_excl_last