
import math
import torch
def get_alibi_slopes(n_heads):
    def get_slopes_2(n):
        start = 2 ** (-2 ** -(math.log2(n) - 3))
        ratio = start
        return [start * (ratio ** i) for i in range(n)]
    if math.log2(n_heads).is_integer():
        return get_slopes_2(n_heads)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
        slopes = get_slopes_2(closest_power_of_2)
        
        extra_slopes = [slopes[-1] * 0.5 ** (i+1) for i in range(n_heads - closest_power_of_2)]
        return slopes + extra_slopes
def get_alibi_bias(n_heads,seq_len, device, dtype, is_decoder=False):
    slopes = torch.tensor(get_alibi_slopes(n_heads), device=device, dtype=dtype)
    positions = torch.arange(seq_len, device=device)
    # distance[i,j] = |i-j|
    distance = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1))
    bias = -slopes.view(-1,1,1) * distance.unsqueeze(0)
    if is_decoder:
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        bias = bias.masked_fill(causal_mask.unsqueeze(0), float('-inf'))
    
    return bias.unsqueeze(0)