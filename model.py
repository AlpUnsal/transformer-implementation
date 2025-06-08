import torch
import torch.nn as nn
import numpy as np

class Encoder(nn.Module):
    def __init__(self, model_dim:int=512, num_heads:int=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=model_dim, num_heads=num_heads)  ### IMPLEMENT FROM SCRATCH
        self.ffnn = nn.Sequential(
            nn.Linear(model_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, model_dim)
        )
        self.layernorm = nn.LayerNorm(normalized_shape=model_dim)

        self.queryM = nn.Parameter(torch.randn(model_dim, model_dim)) 
        self.keyM = nn.Parameter(torch.randn(model_dim, model_dim))
        self.valueM = nn.Parameter(torch.randn(model_dim, model_dim))

    def forward(self, inputs, num_stacks:int=6):
        return self._stack(inputs, num_stacks)

    def _stack(self, input, num_stacks:int=6):
        if num_stacks < 1:
            return input
        
        query = torch.swapaxes(input @ self.queryM, 0, 1)
        key = torch.swapaxes(input @ self.keyM, 0, 1)
        value = torch.swapaxes(input @ self.valueM, 0, 1)

        layer = self.layernorm(input + self.attention.forward(query=query, key=key, value=value))
        
        output = self.layernorm(layer + self.ffnn(layer))

        return self._stack(output, num_stacks-1)    

class Decoder(nn.Module):
    def __init__(self, model_dim:int=512, num_heads:int=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=model_dim, num_heads=num_heads)
        self.maskedAttention = nn.MultiheadAttention(embed_dim=model_dim, num_heads=num_heads)
        self.ffnn = nn.Sequential(
            nn.Linear(model_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, model_dim)
        )
        self.layernorm = nn.LayerNorm(normalized_shape=model_dim)

        self.queryMSelf = nn.Parameter(torch.randn(model_dim, model_dim)) # for self attention layers
        self.keyMSelf = nn.Parameter(torch.randn(model_dim, model_dim))   # ^
        self.valueMSelf = nn.Parameter(torch.randn(model_dim, model_dim)) # ^

        self.queryMDec = nn.Parameter(torch.randn(model_dim, model_dim)) # for 'encoder-decoder' layer
        self.keyMEnc = nn.Parameter(torch.randn(model_dim, model_dim))   # ^
        self.valueMEnc = nn.Parameter(torch.randn(model_dim, model_dim)) # ^

    def forward(self, input, encoder_output, num_stacks:int=6):
        return self._stack(input, encoder_output, num_stacks)

    def _stack(self, input, encoder_output, num_stacks:int=6):
        if num_stacks < 1:
            return input
        
        querySelf = torch.swapaxes(input @ self.queryMSelf, 0, 1)
        keySelf = torch.swapaxes(input @ self.keyMSelf, 0, 1)
        valueSelf = torch.swapaxes(input @ self.valueMSelf, 0, 1)

        queryDec = torch.swapaxes(input @ self.queryMDec, 0, 1)
        keyEnc = torch.swapaxes(encoder_output @ self.keyMEnc, 0, 1)
        valueEnc = torch.swapaxes(encoder_output @ self.valueMEnc, 0, 1)

        
        mask = torch.tril(torch.ones(input.shape[0], input.shape[0]))

        layer1 = self.layernorm(input + self.maskedAttention.forward(query=querySelf, key=keySelf, value=valueSelf, attn_mask=mask))
        layer2 = self.layernorm(layer1 + self.attention.forward(query=queryDec, key=keyEnc, value=valueEnc))

        output = self.layernorm(layer2 + self.ffnn.forward(layer2))

        return self._stack(output, encoder_output, num_stacks-1)

