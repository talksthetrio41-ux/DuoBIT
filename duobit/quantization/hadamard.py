import torch
import math

def hadamard_transform(x: torch.Tensor, scale: bool = True) -> torch.Tensor:
    """
    Applies Fast Walsh-Hadamard Transform (FWHT) along the last dimension of x.
    The last dimension of x must be a power of 2.
    
    x: tensor of shape (..., N) where N = 2^k
    scale: if True, divides by sqrt(N) to make the transformation orthogonal.
    """
    n = x.shape[-1]
    if (n & (n - 1)) != 0 or n == 0:
        raise ValueError(f"Last dimension must be a power of 2, got {n}")

    # Reshape and butterfly operations
    out = x.clone()
    h = 1
    while h < n:
        # Reshape to (-1, n // (2*h), 2, h)
        orig_shape = out.shape
        out = out.view(-1, n // (2 * h), 2, h)
        u = out[:, :, 0, :].clone()
        v = out[:, :, 1, :].clone()
        out[:, :, 0, :] = u + v
        out[:, :, 1, :] = u - v
        out = out.view(orig_shape)
        h *= 2

    if scale:
        out = out / math.sqrt(n)

    return out


class OnlineHadamard(torch.nn.Module):
    """
    Module wrapping tensor inputs with an online Hadamard rotation along the feature dimension.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # Find nearest power of 2 >= dim
        self.padded_dim = 1 << (dim - 1).bit_length()
        self.pad_needed = self.padded_dim - dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad_needed > 0:
            x_padded = torch.nn.functional.pad(x, (0, self.pad_needed))
        else:
            x_padded = x

        x_rot = hadamard_transform(x_padded, scale=True)

        if self.pad_needed > 0:
            return x_rot[..., :self.dim]
        return x_rot
