import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, model_dim:int=512, num_heads:int=8):
        super().__init__()
        # sublayers
        self.attention = nn.MultiheadAttention(embed_dim=model_dim, num_heads=num_heads)  ### IMPLEMENT FROM SCRATCH
        self.ffnn = nn.Sequential(
            nn.Linear(model_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, model_dim)
        )
        self.layernorm = nn.LayerNorm(normalized_shape=model_dim)

        # query, key, value matrices
        self.queryM = nn.Parameter(torch.randn(model_dim, model_dim)) 
        self.keyM = nn.Parameter(torch.randn(model_dim, model_dim))
        self.valueM = nn.Parameter(torch.randn(model_dim, model_dim))

    def forward(self, inputs, num_stacks:int=6):
        """
        Forward pass through the encoder

        Args
            input: (src_len, d) target embeddings
            num_stacks: number of stacks
        """
        return self._stack(inputs, num_stacks)

    def _stack(self, input, num_stacks:int=6):
        """
        Recursive implementation of the stack iteration

        Args
            input: (src_len, d) target embeddings
            num_stacks: number of stacks
        """
        # base case
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

        # sublayers
        self.attention = nn.MultiheadAttention(embed_dim=model_dim, num_heads=num_heads)
        self.maskedAttention = nn.MultiheadAttention(embed_dim=model_dim, num_heads=num_heads)
        self.ffnn = nn.Sequential(
            nn.Linear(model_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, model_dim)
        )
        self.layernorm = nn.LayerNorm(normalized_shape=model_dim)

        # query, key, value matrices for self-attention and encoder-decoder attention
        self.queryMSelf = nn.Parameter(torch.randn(model_dim, model_dim)) # for self attention layers
        self.keyMSelf = nn.Parameter(torch.randn(model_dim, model_dim))   # ^
        self.valueMSelf = nn.Parameter(torch.randn(model_dim, model_dim)) # ^

        self.queryMDec = nn.Parameter(torch.randn(model_dim, model_dim)) # for 'encoder-decoder' layer
        self.keyMEnc = nn.Parameter(torch.randn(model_dim, model_dim))   # ^
        self.valueMEnc = nn.Parameter(torch.randn(model_dim, model_dim)) # ^

    def forward(self, input, encoder_output, num_stacks:int=6):
        """
        Forward pass through the decoder

        Args
            input: (tgt_len, d) target embeddings
            encoder_output: output from the encoder to be joined in the attention head
            num_stacks: number of stacks
        """
        return self._stack(input, encoder_output, num_stacks)

    def _stack(self, input, encoder_output, num_stacks:int=6):
        """
        Recursive implementation of the stack iteration

        Args
            input: (tgt_len, d) target embeddings
            encoder_output: output from the encoder to be joined in the attention head
            num_stacks: number of stacks
        """

        if num_stacks < 1:
            return input
        
        querySelf = torch.swapaxes(input @ self.queryMSelf, 0, 1)
        keySelf = torch.swapaxes(input @ self.keyMSelf, 0, 1)
        valueSelf = torch.swapaxes(input @ self.valueMSelf, 0, 1)

        queryDec = torch.swapaxes(input @ self.queryMDec, 0, 1)
        keyEnc = torch.swapaxes(encoder_output @ self.keyMEnc, 0, 1)
        valueEnc = torch.swapaxes(encoder_output @ self.valueMEnc, 0, 1)

        # mask
        mask = torch.tril(torch.ones(input.shape[0], input.shape[0], dtype=torch.bool))

        layer1 = self.layernorm(input + self.maskedAttention.forward(query=querySelf, key=keySelf, value=valueSelf, attn_mask=mask))
        layer2 = self.layernorm(layer1 + self.attention.forward(query=queryDec, key=keyEnc, value=valueEnc))

        output = self.layernorm(layer2 + self.ffnn.forward(layer2))

        return self._stack(output, encoder_output, num_stacks-1)


class Transformer(nn.Module):
    def __init__(self, model_dim:int=512, num_heads:int=8, max_seq_len:int=5000, vocab_size:int=10000):
        super().__init__()

        # layers
        self.encoder = Encoder(model_dim=model_dim, num_heads=num_heads)
        self.decoder = Decoder(model_dim=model_dim, num_heads=num_heads)

        self.linear = nn.Linear(model_dim, vocab_size)
        self.softmax = nn.Softmax(dim=1)

        # embedding
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=model_dim)

        # positional encoding
        pos = torch.unsqueeze(torch.arange(max_seq_len), 1)
        dim = torch.unsqueeze(torch.arange(model_dim // 2), 0)

        denom = 10000 ** (2 * dim / model_dim)
        even = torch.sin(pos / denom)
        odd = torch.cos(pos / denom)

        posEncoding = torch.zeros((max_seq_len, model_dim))
        posEncoding[:, 0::2] = even
        posEncoding[:, 1::2] = odd

        self.register_buffer('posEncoding', posEncoding)

    def forward(self, source, target, num_stacks:int=6):
        """
        Forward pass through the entire architecture
        
        Args
            source: (src_len,)   LongTensor of token ids
            target: (tgt_len,)   LongTensor of token ids
        """
        sourceE = self.embedding(source)
        targetE = self.embedding(target)

        # positional encoding
        x = sourceE + self.posEncoding[:len(source)].to(sourceE.device)
        y = targetE + self.posEncoding[:len(target)].to(targetE.device)

        # encoder
        x = self.encoder.forward(torch.unsqueeze(x, 1), num_stacks=num_stacks)

        # beginning of sentence tensor
        bos = self.embedding(torch.tensor([1], device=target.device)) # using an abritrary value for BoS token value

        # moving over one position before putting into decoder
        y = torch.cat((bos, y[:-1]), dim=0)
        y = self.decoder.forward(y.unsqueeze(1), encoder_output=x, num_stacks=num_stacks)

        # linear + softmax before output
        out = self.linear(y).squeeze(1)
        out = self.softmax(out)

        return out