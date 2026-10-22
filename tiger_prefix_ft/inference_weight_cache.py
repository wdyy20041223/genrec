"""Cache the FP16 casts already performed by autocast, keeping FP32 parameters."""
import torch
from torch import nn
from torch.nn import functional as F


class FrozenLinearCastCache:
    """Optional frozen-inference context; normalization/embedding weights stay FP32.

    Each cached value is exactly weight.to(float16), as used by CUDA autocast
    for Linear. The original Parameter objects, storage and dtype are retained.
    Close this context before changing checkpoints, device or training mode.
    """
    def __init__(self, model):
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError('Linear cast caching requires a frozen eval model')
        self.saved, self.bytes = [], 0
        layers = [layer for layer in model.modules() if isinstance(layer, nn.Linear)]
        if any(layer.weight.device.type != 'cuda' or layer.weight.dtype != torch.float32 for layer in layers):
            raise ValueError('Expected the original CUDA FP32 parameters')
        prepared = []
        for layer in layers:
            weight = layer.weight.detach().to(torch.float16)
            bias = None if layer.bias is None else layer.bias.detach().to(torch.float16)
            prepared.append((layer, weight, bias))
        for layer, weight, bias in prepared:
            self.bytes += weight.numel()*weight.element_size()
            if bias is not None:
                self.bytes += bias.numel()*bias.element_size()
            original = layer.forward
            def forward(input, _weight=weight, _bias=bias, _original=original):
                if torch.is_autocast_enabled('cuda') and torch.get_autocast_dtype('cuda') == torch.float16:
                    return F.linear(input, _weight, _bias)
                return _original(input)
            self.saved.append((layer, original, layer.weight.data_ptr(), layer.weight.dtype))
            layer.forward = forward

    def __enter__(self):
        return self

    def close(self):
        for layer, original, address, dtype in self.saved:
            layer.forward = original
            assert layer.weight.data_ptr() == address and layer.weight.dtype == dtype
        self.saved.clear()

    def __exit__(self, *exc):
        self.close()
