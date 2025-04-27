import torch
import torch.nn as nn
from timm.layers import LayerNorm2d


class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, channels_last, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}, channels_last={self.channels_last}"
    

class M_DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, channels_last, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last
        
        # Get the last dimension for the linear layer
        self.linear = nn.Linear(normalized_shape[-1], 1)
        
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):       
        if self.channels_last:
            M = self.linear(x)  # [B, 1]
            M = M.expand_as(x)  # [B, D]
            x = torch.tanh(self.alpha * M * x)
            x = x * self.weight + self.bias
        else:
            x = torch.tanh(self.alpha * x)
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
       
        return x

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}, channels_last={self.channels_last}"

class M_DynamicTanh_plus(nn.Module):
    def __init__(self, normalized_shape, channels_last, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last
        
        # Get the last dimension for the linear layer
        self.linear = nn.Linear(normalized_shape[-1], 1)
        
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):       
        if self.channels_last:           
            M = self.linear(x)  # [B, 1]
            M_plus = M.expand_as(x) + torch.ones(x.size()).to(x.device)  # [B, D]
            x = torch.tanh(self.alpha * M_plus * x)
            x = x * self.weight + self.bias
        else:
            x = torch.tanh(self.alpha * x)
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
       
        return x

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}, channels_last={self.channels_last}"


class M_DynamicTanh_mean(nn.Module):
    def __init__(self, normalized_shape, channels_last, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last
                
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):       
        if self.channels_last:           
            M = x.mean(dim=1).unsqueeze(1)  # [B, 1]
            M_mean = M.expand_as(x) + torch.ones(x.size()).to(x.device)  # [B, D]
            x = torch.tanh(self.alpha * M_mean * x)
            x = x * self.weight + self.bias
        else:
            x = torch.tanh(self.alpha * x)
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
       
        return x

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}, channels_last={self.channels_last}"

def convert_ln_to_m_dyt_mean(module):
    module_output = module
    if isinstance(module, nn.LayerNorm):
        module_output = M_DynamicTanh_mean(module.normalized_shape, not isinstance(module, LayerNorm2d))
    for name, child in module.named_children():
        module_output.add_module(name, convert_ln_to_m_dyt_mean(child))
    del module
    return module_output


def convert_ln_to_m_dyt_plus(module):
    module_output = module
    if isinstance(module, nn.LayerNorm):
        module_output = M_DynamicTanh_mean(module.normalized_shape, not isinstance(module, LayerNorm2d))
    for name, child in module.named_children():
        module_output.add_module(name, convert_ln_to_m_dyt_plus(child))
    del module
    return module_output


def convert_ln_to_m_dyt(module):
    module_output = module
    if isinstance(module, nn.LayerNorm):
        module_output = M_DynamicTanh(module.normalized_shape, not isinstance(module, LayerNorm2d))
    for name, child in module.named_children():
        module_output.add_module(name, convert_ln_to_m_dyt(child))
    del module
    return module_output


def convert_ln_to_dyt(module):
    module_output = module
    if isinstance(module, nn.LayerNorm):
        module_output = DynamicTanh(module.normalized_shape, not isinstance(module, LayerNorm2d))
    for name, child in module.named_children():
        module_output.add_module(name, convert_ln_to_dyt(child))
    del module
    return module_output

