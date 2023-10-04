import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableAlpha(nn.Module):
    # GeLU layer with learnable parameter alpha. 
    def __init__(self,config):
        super(LearnableAlpha, self).__init__()
        self.patches = int((config.image_size*config.image_size)/(config.patch_size*config.patch_size)+1)
        self.intermediate_size = config.intermediate_size
        # self.alphas = nn.Parameter(torch.ones((1, self.patches, self.intermediate_size), requires_grad=True)) # pointwise replacement of gelus
        self.alphas = nn.Parameter(torch.ones((1, self.patches, 1), requires_grad=True)) # token wise replacement of gelus

    def forward(self, x):
        out = F.gelu(x) * self.alphas.expand_as(x) + (1-self.alphas.expand_as(x)) * x 
        return out

class LearnableBeta(nn.Module):
    # Attn mask heads with learnable parameter beta. 
    def __init__(self, config):
        super(LearnableBeta, self).__init__()
        self.patches = int((config.image_size*config.image_size)/(config.patch_size*config.patch_size)+1)
        self.attention_heads = config.num_attention_heads
        self.betas = nn.Parameter(torch.ones((1, self.attention_heads, self.patches,1), requires_grad=True))

    def forward(self, x):
        out = F.softmax(x,dim=-1) * self.betas.expand_as(x) + (1-self.betas.expand_as(x)) * (torch.square(x)/197) 
        return out