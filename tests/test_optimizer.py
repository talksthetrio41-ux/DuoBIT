import torch
import torch.nn as nn
import pytest
from duobit.layers.duobit_linear import DuobitLinear
from duobit.optim.duobit_adam import DuobitAdam

def test_duobit_adam_single_layer():
    torch.manual_seed(42)
    layer = DuobitLinear(in_features=128, out_features=64, group_size=128, use_hadamard=False)
    
    # Dummy container module
    class SimpleModel(nn.Module):
        def __init__(self, lin):
            super().__init__()
            self.lin = lin
        def forward(self, x):
            return self.lin(x)
            
    model = SimpleModel(layer)
    optimizer = DuobitAdam(model, lr=0.01)
    
    initial_codes = layer.codes.clone()
    
    # Run 10 forward-backward-step iterations
    x = torch.randn(16, 128)
    for _ in range(10):
        optimizer.zero_grad()
        out = model(x)
        loss = out.pow(2).mean()
        loss.backward()
        optimizer.step()
        
    # Check that codes have changed over 10 steps
    diff_codes = (layer.codes != initial_codes).sum().item()
    assert diff_codes > 0, "Expected some 2-bit weight code transitions after 10 optimizer steps"

def test_duobit_adam_with_standard_params():
    torch.manual_seed(42)
    
    class ModelWithMixedParams(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(128)
            self.duobit = DuobitLinear(in_features=128, out_features=128, group_size=128)
            
        def forward(self, x):
            return self.duobit(self.norm(x))
            
    model = ModelWithMixedParams()
    optimizer = DuobitAdam(model, lr=0.01)
    
    x = torch.randn(8, 128)
    out = model(x)
    loss = out.sum()
    loss.backward()
    
    initial_norm_weight = model.norm.weight.clone()
    optimizer.step()
    
    # Standard parameter (LayerNorm weight) should update via standard Adam
    assert not torch.allclose(model.norm.weight, initial_norm_weight)
